"""Who said each settled line: Deepgram's word-level speaker labels first, voiceprints as a careful check.

Deepgram's labels ("speaker 0", "speaker 1") mean something only inside one streaming connection. Each
(connection, label) is mapped to a person of this conversation; a label never inherits a person just
because the number is reused by another connection.

Voiceprints (SpeechBrain ECAPA, on this laptop, ~50 ms a sentence) may only:
  * recognise a NEW label as a person already heard, from at least MIN_NEW seconds of speech;
  * split a long line where the voice clearly changes inside it (someone cut in, and Deepgram's live
    labels missed it), and attribute each part only if the part clearly sounds like someone;
  * mark a line "uncertain" (no speaker) when they clearly disagree with Deepgram's label.
They never quietly overrule Deepgram on a short clip: short clips give unreliable voiceprints (seen live on
14 Nov: 0.3-0.7 s clips moved correct Deepgram labels to the wrong person).

A person's voiceprint learns only from clear single-speaker audio: unsplit lines of at least MIN_LEARN
seconds where voiceprint and label agree, or split parts of at least 1.5 s, with the edges (where another
voice may bleed in) trimmed off. Uncertain lines never teach anything.

Privacy: the sound (last 90 s) and the voiceprints stay in memory for this conversation only.
Needs: torch, speechbrain, av (pip). Without them, Deepgram's labels are used as they are.
"""
import os
import queue
import sys
import threading
import time

import numpy as np

SAMPLE_RATE = 16000
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_DIR = os.path.join(APP_DIR, "models", "ecapa")

# Who said it. Tune these from /debug (each line's similarities are listed there) and eval/run.py.
DISAGREE = "uncertain"   # voiceprint clearly against Deepgram's label: "uncertain" (no speaker) or "voice" (move it)
REIDENTIFY = 0.50        # a new Deepgram label is a known person if its first long line is this similar
SAME_VOICE = 0.40        # similarity to a person's voiceprint at or above this: it is them
                         # (real speech on the band, 14 Nov: same person ~0.4-0.65, different people ~0.0-0.25)
NEW_VOICE = 0.20         # below this for EVERY person (and the sentence is long enough): a new person
MIN_PRINT = 0.6          # seconds: shorter sentences ("Priya.") get a voiceprint but count as weak evidence
MIN_NEW = 1.5            # seconds: a new person is only started from a sentence at least this long
MIN_LEARN = 0.8          # seconds: only sentences this long update a person's average voiceprint
# Same-person suggestions between two people, and the wearer
MIN_VOICE = 2.5          # seconds of speech a person needs before being compared with another
SAME_LOW = 0.30          # similarity that maps to confidence 0 ...
SAME_HIGH = 0.65         # ... and to confidence 1. So 0.60+ -> join, 0.48-0.60 -> ask "Same person?"
WEARER_DB = 6.0          # the wearer's typical sentence is this much louder than everyone else's
SHORT_DB = 10.0          # a short sentence this much quieter or louder than Deepgram's pick is someone else
# Someone cutting in: one Deepgram sentence that holds two voices is split where the voice changes
SPLIT_MIN = 2.0          # seconds: only sentences at least this long are checked
WINDOW, HOP = 1.2, 0.4   # seconds: a voiceprint of each 1.2 s piece, every 0.4 s
MIN_PART = 0.7           # seconds: each part of a split must be at least this long
BUFFER_SECONDS = 90      # how much recent sound is kept in memory to cut sentences from
# The wearer's enrolled baseline (familiar/enroll.py): a fixed voiceprint for person ME, never updated live
ME = 0
ME_MIN = 1.0             # seconds of speech before a line is compared with the baseline


def load_encoder(model_dir=MODEL_DIR):
    """A function: float32 samples at 16 kHz -> unit-length voiceprint. Downloads the model once."""
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    model = EncoderClassifier.from_hparams(source=MODEL_SOURCE, savedir=model_dir, run_opts={"device": "cpu"})

    def encode(samples):
        with torch.no_grad():
            vector = model.encode_batch(torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))[None])
        vector = vector.squeeze().numpy().astype(np.float32)
        return vector / (np.linalg.norm(vector) or 1.0)

    def many(pieces):
        """Several pieces of the same length in one pass (the windows inside a sentence)."""
        with torch.no_grad():
            vectors = model.encode_batch(torch.from_numpy(np.stack(pieces).astype(np.float32))).squeeze(1).numpy()
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    encode.many = many
    return encode


def loudness_db(samples):
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if len(samples) else 0.0
    return 20.0 * np.log10(max(rms, 1e-6))


class VoicePrints:
    """add_audio() with the band's sound as it arrives; settle() with each settled line from Deepgram;
    interim() with Deepgram's provisional text. The work runs on its own thread: nothing here waits."""

    def __init__(self, resolver, encoder=None, loader=load_encoder, log=None, enabled=True, me=None):
        self.resolver = resolver
        self.me = me                            # the enrolled baseline profile (with "vector", "threshold") or None
        self.encode = encoder
        self._loader = loader if (encoder is None and enabled) else None
        self.log = log if log is not None else (lambda text: print(text, file=sys.stderr))
        self._lock = threading.Lock()
        self._buffer = np.zeros(BUFFER_SECONDS * SAMPLE_RATE, dtype=np.float32)
        self._written = 0                       # samples received since the stream started
        self._jobs = queue.Queue(maxsize=200)
        self._thread = None
        self.ready = threading.Event()          # the speaker model is loaded
        self.failed = encoder is None and not enabled   # voice matching off: Deepgram's labels as they are
        self.on_segment = None                  # called with each settled line (the second pass listens here)
        self._clear()

    def _clear(self):
        self.voices = {}                        # person id -> voiceprint {"sum", "seconds", "loud"}: clean speech only
        if self.me is not None:                 # the wearer, known from enrolment before anyone speaks
            self.voices[ME] = {"sum": np.asarray(self.me["vector"], dtype=np.float32) * 100.0,
                               "seconds": max(100.0, float(self.me.get("seconds", 0))), "loud": [], "fixed": True}
        self.label_person = {}                  # (connection, Deepgram label) -> person id
        self.last = None                        # person of the last attributed line
        self.decisions = []                     # the last few choices, for /debug
        self.epoch = self.resolver.epoch

    # ------------------------------------------------------------------ input (any thread, never blocks)

    def add_audio(self, pcm):
        """16-bit mono PCM bytes at 16 kHz, in the same order Deepgram gets the packets."""
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        total = len(samples)
        with self._lock:
            size = len(self._buffer)
            if total > size:
                samples = samples[-size:]
            n = len(samples)
            start = (self._written + total - n) % size
            first = min(n, size - start)
            self._buffer[start:start + first] = samples[:first]
            self._buffer[:n - first] = samples[first:]
            self._written += total

    def settle(self, label, text, start, end, words=None, connection=0):
        """A settled line from Deepgram: its speaker label, text, audio start/end (seconds since the stream
        began), its words [(word, start, end), ...], and which streaming connection it came from."""
        try:
            self._jobs.put_nowait((label, text, start, end, words, connection))
        except queue.Full:                      # far behind: hand it on under Deepgram's label
            self._emit(self.person_for(connection, label), text, start, end, words, "provider", [], label,
                       connection, "falling behind: Deepgram's label")

    def person_for(self, connection, label, create=True):
        """The person a (connection, label) stands for in this conversation (merges followed)."""
        if label is None:
            return self.last
        key = (connection, label)
        person = self.label_person.get(key)
        if person is None:
            if not create:
                return None
            person = self.label_person[key] = self._new_id()
        return self.resolver.alias.get(person, person)

    def interim(self, pieces, connection=0):
        """Deepgram's provisional text for the page, REPLACING the previous: [(label, text), ...].
        A label not seen in a settled line yet shows as a new voice, without making a card."""
        self.resolver.set_interim([{"speaker": self.person_for(connection, label, create=False), "text": text}
                                   for label, text in pieces])

    def cut(self, start, end):
        """The samples between start and end (seconds), or None if they are not (or no longer) in memory."""
        a, b = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        with self._lock:
            size = len(self._buffer)
            if b > self._written or a < self._written - size or b <= a:
                return None
            return self._buffer[np.arange(a, b) % size].copy()

    # ------------------------------------------------------------------ the worker

    def start(self):
        if self.encode is None and self._loader is not None:
            threading.Thread(target=self._load, name="voice-model", daemon=True).start()
        else:
            self.ready.set()
        self._thread = threading.Thread(target=self._run, name="voiceprints", daemon=True)
        self._thread.start()
        return self

    def _load(self):
        try:
            self.encode = self._loader()
            self.ready.set()
        except Exception as problem:                 # missing package, no internet for the first download...
            self.failed = True
            self.log("Voice matching is off: %s" % (str(problem).splitlines()[0][:160] if str(problem) else
                                                    type(problem).__name__))

    def stop(self):
        self._jobs.put(None)

    def _run(self):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                self.process(*job)
            except Exception as problem:                 # one bad line must not stop the rest
                self.log("Voice matching skipped a line: %s" % type(problem).__name__)
                label, text, start, end, words, connection = job
                self._emit(self.person_for(connection, label), text, start, end, words, "provider", [], label,
                           connection, "error: Deepgram's label")

    def _sound(self, start, end):
        """The samples of [start, end], waiting briefly if the words arrived before their sound."""
        if not self.ready.is_set() or start is None or end is None or end <= start:
            return None
        for _ in range(20):
            samples = self.cut(start, end)
            if samples is not None or self._written / SAMPLE_RATE > end:
                return samples
            time.sleep(0.05)
        return None

    def process(self, label, text, start, end, words=None, connection=0):
        """Attribute one settled line (or each part of it, if the voice changes inside it) and hand it on."""
        if self.resolver.epoch != self.epoch:            # "Start a new conversation": forget everyone
            self._clear()
        words = [tuple(w) for w in words or []]
        samples = self._sound(start, end)
        base = self.person_for(connection, label, create=False)
        first = None
        if base is None and label is not None:
            base, first = self._first_line_of_label(connection, label, samples, end - start if samples is not None else 0)
        parts = self.split(words, start, end) if samples is not None and words else None
        if parts and len(parts) >= 2:
            used = set()
            for part in parts:
                self._part(base, label, connection, part, used)
        else:
            self._whole(base, label, connection, text, start, end, words, samples, first)
        if self.ready.is_set():                          # after learning: who sounds alike, who is loudest
            self.resolver.diagnostics["voices"] = dict(self.stats(), decisions=self.decisions)
            self.resolver.apply_model_opinion(self.opinion(), source="voice")

    def _sounds_like_me(self, samples, seconds):
        """Similarity to the enrolled baseline, or None (no baseline, no sound, too short)."""
        if self.me is None or samples is None or seconds < ME_MIN:
            return None
        return float(self.encode(samples) @ self.voices[ME]["sum"] / np.linalg.norm(self.voices[ME]["sum"]))

    def _claim_me(self, key, person):
        """This label is the wearer: map it to ME (merging a provisional person) and mark ME as "you"."""
        if person is not None and person != ME and person in self.resolver.people:
            with self.resolver._lock:
                self.resolver._person(ME)               # "You" must exist before the provisional person joins it
            self.resolver.merge(person, ME)
        self.label_person[key] = ME
        if person not in (None, ME):
            self.voices.pop(person, None)

    def _first_line_of_label(self, connection, label, samples, seconds):
        """A label never seen before: the wearer if it matches the baseline, a known person if its voice clearly
        matches one, else a new person."""
        key = (connection, label)
        mine = self._sounds_like_me(samples, seconds)
        taken_me = any(p == ME for (c, _), p in self.label_person.items() if c == connection)
        if mine is not None and mine >= self.me["threshold"] + (0.15 if taken_me else 0.0):
            self._claim_me(key, None)
            return ME, "sounds like you (baseline %.2f >= %.2f)" % (mine, self.me["threshold"])
        profiles = {k: g for k, g in self.grouped().items() if g[1] >= MIN_VOICE}
        if samples is not None and seconds >= MIN_NEW and profiles:
            vector = self.encode(samples)
            sims = {k: float(g[0] @ vector) for k, g in profiles.items()}
            best = max(sims, key=sims.get)
            taken = {self.resolver.alias.get(p, p) for (c, lab), p in self.label_person.items() if c == connection}
            if sims[best] >= REIDENTIFY and best not in taken:
                self.label_person[key] = best
                return best, "new label, sounds like a person already heard (%.2f)" % sims[best]
        self.label_person[key] = self._new_id()
        return self.label_person[key], None

    def _whole(self, base, label, connection, text, start, end, words, samples, first):
        seconds = (end - start) if samples is not None else 0.0
        status, who, candidates, sims = ("voice" if first else "provider"), base, [], {}
        why = first or "Deepgram's label"
        if self.me is not None and base != ME and samples is not None:
            # a label first heard on a line too short to check may be the wearer: claim it once it clearly is
            young = self.voices.get(base, {}).get("seconds", 0) < MIN_VOICE
            mine = self._sounds_like_me(samples, seconds)
            key = (connection, label)
            taken_me = any(p == ME for k, p in self.label_person.items() if k[0] == connection and k != key)
            need = self.me["threshold"] + (0.15 if taken_me else 0.05)
            if young and mine is not None and mine >= need and seconds >= MIN_NEW:
                self._claim_me(key, base)
                base, who, status = ME, ME, "voice"
                why = "this label is you (baseline %.2f)" % mine
        vector = loud = None
        if samples is not None and seconds >= MIN_PRINT:
            vector, loud = self.encode(samples), loudness_db(samples)
            profiles = {k: g for k, g in self.grouped().items() if g[1] > 0}
            sims = {k: float(g[0] @ vector) for k, g in profiles.items()}
            mine = profiles.get(base)
            others = {k: v for k, v in sims.items() if k != base}
            other = max(others, key=others.get) if others else None
            if mine is not None and mine[1] >= MIN_VOICE and sims[base] < NEW_VOICE and seconds >= MIN_NEW:
                if other is not None and others[other] >= SAME_VOICE:
                    if DISAGREE == "voice":
                        who, status, why = other, "voice", "voiceprint clearly someone else than Deepgram's label"
                    else:
                        who, status, candidates = None, "uncertain", [base, other]
                        why = "voiceprint and Deepgram's label disagree: left unresolved"
                else:   # Deepgram may have lumped a new voice in with this label: do not give it these words
                    who, status, candidates = None, "uncertain", [base]
                    why = "voiceprint unlike this label's voice, and like nobody else: left unresolved"
        self._emit(who, text, start, end, words, status, candidates, label, connection, why, seconds, loud, sims)
        agrees = who == base and (base not in sims or sims[base] >= NEW_VOICE or self.voices.get(base, {}).get("seconds", 0) < MIN_VOICE)
        if who is not None and agrees and seconds >= MIN_LEARN:
            self._learn(who, start, end, trim=0.15)

    def _part(self, base, label, connection, part, used):
        """One part of a line that held two voices: attributed only if it clearly sounds like someone."""
        start, end = part[0][1], part[-1][2]
        text = " ".join(w for w, _, _ in part)
        samples = self._sound(start, end)
        seconds = end - start
        who, status, candidates, sims, loud = None, "uncertain", [], {}, None
        why = "part of a line that held two voices: unclear who"
        if samples is not None and seconds >= MIN_PRINT * 0.5:
            vector, loud = self.encode(samples), loudness_db(samples)
            profiles = {k: g for k, g in self.grouped().items() if g[1] > 0}
            sims = {k: float(g[0] @ vector) for k, g in profiles.items()}
            best = max(sims, key=sims.get) if sims else None
            if best is not None and sims[best] >= SAME_VOICE:
                who, status, why = best, "voice", "part of a split line: sounds like them"
            elif (not sims or sims[best] < NEW_VOICE) and seconds >= MIN_NEW:
                if base is not None and base not in used and self.voices.get(base, {}).get("seconds", 0) == 0:
                    who, status, why = base, "voice", "part of a split line: the label's own first voice"
                else:
                    who, status, why = self._new_id(), "voice", "part of a split line: a voice not heard before"
            else:
                candidates = sorted(k for k, v in sims.items() if v >= NEW_VOICE)
        if who is not None:
            used.add(who)
        self._emit(who, text, start, end, part, status, candidates, label, connection, why, seconds, loud, sims,
                   split=True)
        if who is not None and seconds >= MIN_NEW:
            self._learn(who, start, end, trim=0.3)

    def _learn(self, who, start, end, trim):
        """Add clear single-speaker audio to a person's voiceprint (edges trimmed: another voice may bleed in).
        The enrolled baseline is fixed: live audio never changes it."""
        if self.voices.get(who, {}).get("fixed"):
            return
        piece = self.cut(start + trim, end - trim) if end - start > 2 * trim + MIN_PRINT else None
        if piece is None or not len(piece):
            return
        vector = self.encode(piece)
        seconds = len(piece) / SAMPLE_RATE
        v = self.voices.setdefault(who, {"sum": np.zeros_like(vector), "seconds": 0.0, "loud": []})
        v["sum"] = v["sum"] + vector * seconds
        v["seconds"] += seconds
        v["loud"] = (v["loud"] + [loudness_db(piece)])[-30:]

    def _emit(self, who, text, start, end, words, status, candidates, label, connection, why,
              seconds=None, loud=None, sims=None, split=False):
        meta = {"start": start, "end": end, "words": [list(w) for w in words or []], "status": status,
                "candidates": [c for c in candidates if c is not None], "label": "%s:%s" % (connection, label)}
        self.resolver.feed(who, text, final=True, meta=meta)
        if who == ME and self.me is not None and not self.resolver.people.get(ME, {}).get("is_me"):
            self.resolver.set_me(ME)                     # the enrolled wearer is "You"
        if who is not None:
            self.last = who
        self.decisions = (self.decisions + [{
            "text": text[:40], "deepgram": label, "connection": connection, "person": who, "status": status,
            "seconds": round(seconds, 1) if seconds else None, "loudness_db": round(loud, 1) if loud else None,
            "similarity": {str(k): round(v, 2) for k, v in (sims or {}).items()}, "why": why}])[-14:]
        self.resolver.diagnostics["voices"] = dict(self.stats(), decisions=self.decisions)
        if self.on_segment:
            line = self.resolver.log[-1] if self.resolver.log else None
            try:
                self.on_segment(line)
            except Exception as problem:                 # the second pass must never stop the live path
                self.log("Second pass: %s" % type(problem).__name__)

    def _new_id(self):
        taken = (set(self.resolver.people) | set(self.resolver.alias) | set(self.voices)
                 | set(self.label_person.values()))
        n = 0
        while n in taken:
            n += 1
        return n

    def split(self, words, start, end):
        """[[words of part 1], [words of part 2], ...] where the voice changes inside one sentence, or None.

        Two clues: a voiceprint of every 1.2 s window (which known person, or nobody known, does it sound
        like?) and the loudness of each word (the wearer is much louder than everyone else)."""
        words = [w for w in words if w[2] > w[1]]
        if len(words) < 4 or end is None or start is None or end - start < SPLIT_MIN:
            return None
        if self.cut(start, end) is None:
            return None
        labels = self._window_labels(words, start, end)
        runs = self._runs(words, labels)
        named = [lab for lab in labels if lab is not None]
        agreed = bool(named) and named.count(max(set(named), key=named.count)) >= 0.8 * len(labels)
        if len(runs) < 2 and not agreed:                # the voiceprints could not tell: try loudness
            runs = self._loudness_split(words)
        if not runs or len(runs) < 2:
            return None
        runs = self._confirm(runs)                      # neighbours that sound alike are glued back
        if len(runs) < 2:
            return None
        for i in range(len(runs) - 1):                  # move each cut to the exact word
            runs[i], runs[i + 1] = self._refine(runs[i], runs[i + 1])
        return [part for part in runs if part]

    def _refine(self, left, right, reach=2):
        """The 1.2 s windows put a cut within a word or two of the change; look at the words near it one by
        one (by loudness when the two voices differ a lot, else by a short voiceprint around each word) and
        move the cut to where the words agree best."""
        words = left + right
        k = len(left)
        lo, hi = max(1, k - reach), min(len(words) - 1, k + reach)
        core_l, core_r = words[:max(1, lo - 1)], words[min(len(words) - 1, hi + 1):]
        if not core_l or not core_r:
            return left, right
        def loud_of(ws):
            piece = self.cut(ws[0][1], ws[-1][2])
            return loudness_db(piece) if piece is not None and len(piece) else None
        word_loud = [loud_of([w]) for w in words]
        ref_l, ref_r = loud_of(core_l), loud_of(core_r)
        side = {}
        if None not in (ref_l, ref_r) and abs(ref_l - ref_r) >= WEARER_DB / 2:
            for j in range(lo - 1, hi + 1):
                if word_loud[j] is not None:
                    side[j] = "L" if abs(word_loud[j] - ref_l) < abs(word_loud[j] - ref_r) else "R"
        else:
            a = self.cut(core_l[0][1], core_l[-1][2])
            b = self.cut(core_r[0][1], core_r[-1][2])
            if a is None or b is None or not len(a) or not len(b):
                return left, right
            v_l, v_r = self.encode(a), self.encode(b)
            for j in range(lo - 1, hi + 1):
                mid = (words[j][1] + words[j][2]) / 2
                piece = self.cut(mid - 0.4, mid + 0.4)
                if piece is not None and len(piece):
                    v = self.encode(piece)
                    side[j] = "L" if float(v @ v_l) > float(v @ v_r) else "R"
        if not side:
            return left, right
        best = min(range(lo, hi + 1), key=lambda c: (sum(1 for j, sd in side.items() if (j < c) != (sd == "L")),
                                                     abs(c - k)))
        return words[:best], words[best:]

    def _confirm(self, runs):
        """Keep a split only where the two neighbouring parts really sound different."""
        out = [runs[0]]
        for part in runs[1:]:
            a, b = self.cut(out[-1][0][1], out[-1][-1][2]), self.cut(part[0][1], part[-1][2])
            if a is not None and b is not None and len(a) and len(b) and \
                    float(self.encode(a) @ self.encode(b)) >= SAME_VOICE:
                out[-1] = out[-1] + part
            else:
                out.append(part)
        return out

    def _window_labels(self, words, start, end):
        """For each word: the known person its windows sound like, "new" if like nobody known, or None."""
        known = {k: g for k, g in self.grouped().items() if g[1] > 0}
        starts = np.arange(start, max(start, end - WINDOW) + 1e-6, HOP)
        pieces, spans = [], []
        for s0 in starts:
            piece = self.cut(s0, s0 + WINDOW)
            if piece is not None and len(piece) == int(WINDOW * SAMPLE_RATE):
                pieces.append(piece)
                spans.append((s0, s0 + WINDOW))
        if len(pieces) < 2:
            return [None] * len(words)
        many = getattr(self.encode, "many", None)
        vectors = many(pieces) if many else np.stack([self.encode(p) for p in pieces])
        window_label = []
        for v in vectors:
            sims = {k: float(g[0] @ v) for k, g in known.items()}
            best = max(sims, key=sims.get) if sims else None
            if best is not None and sims[best] >= SAME_VOICE - 0.05:
                window_label.append(best)
            elif not sims or sims[best] < NEW_VOICE:
                window_label.append("new")
            else:
                window_label.append(None)
        if not known:                                   # nobody known yet: compare the windows with each other
            first = vectors[0]
            window_label = ["a" if float(first @ v) >= SAME_VOICE - 0.05 else
                            ("b" if float(first @ v) < NEW_VOICE else None) for v in vectors]
        labels = []
        for _, w0, w1 in words:
            mid = (w0 + w1) / 2
            votes = [lab for lab, (a, b) in zip(window_label, spans) if a <= mid <= b and lab is not None]
            labels.append(max(set(votes), key=votes.count) if votes else None)
        return labels

    @staticmethod
    def _runs(words, labels):
        """Words grouped into runs of one label; runs shorter than MIN_PART join their neighbour."""
        runs = []
        for word, label in zip(words, labels):
            if runs and (label is None or label == runs[-1][0] or runs[-1][0] is None):
                if runs[-1][0] is None:
                    runs[-1][0] = label
                runs[-1][1].append(word)
            else:
                runs.append([label, [word]])
        merged = []
        for label, part in runs:
            if merged and (part[-1][2] - part[0][1] < MIN_PART):
                merged[-1][1].extend(part)
            else:
                merged.append([label, part])
        if len(merged) > 1 and merged[0][1][-1][2] - merged[0][1][0][1] < MIN_PART:
            merged[1][1][:0] = merged[0][1]
            merged.pop(0)
        out = []
        for label, part in merged:                      # neighbours that ended up with the same label: one part
            if out and out[-1][0] == label:
                out[-1][1].extend(part)
            else:
                out.append([label, part])
        return [part for _, part in out]

    def _loudness_split(self, words):
        """Two parts where the loudness jumps by WEARER_DB or more (the wearer cutting in, or the reverse)."""
        loud = []
        for _, w0, w1 in words:
            piece = self.cut(w0, w1)
            loud.append(loudness_db(piece) if piece is not None and len(piece) else None)
        if any(x is None for x in loud):
            return None
        best, best_gap = None, 0.0
        for k in range(1, len(words)):
            left, right = words[:k], words[k:]
            if left[-1][2] - left[0][1] < MIN_PART or right[-1][2] - right[0][1] < MIN_PART:
                continue
            gap = abs(float(np.median(loud[:k])) - float(np.median(loud[k:])))
            if gap > best_gap:
                best, best_gap = k, gap
        if best is None or best_gap < WEARER_DB:
            return None
        return [words[:best], words[best:]]

    # ------------------------------------------------------------------ what the voices say

    def grouped(self):
        """People as the Resolver sees them now (after merges): id -> (unit voiceprint, seconds, loudness list)."""
        groups = {}
        alias = dict(self.resolver.alias)
        for voice, v in self.voices.items():
            key = alias.get(voice, voice)
            g = groups.setdefault(key, {"sum": np.zeros_like(v["sum"]), "seconds": 0.0, "loud": []})
            g["sum"] = g["sum"] + v["sum"]
            g["seconds"] += v["seconds"]
            g["loud"] += v["loud"]
        out = {}
        for key, g in groups.items():
            norm = np.linalg.norm(g["sum"])
            out[key] = (g["sum"] / norm if norm else g["sum"], g["seconds"], g["loud"])
        return out

    def stats(self):
        """Numbers for /debug: seconds of speech, typical loudness, and how alike each pair sounds."""
        groups = self.grouped()
        keys = sorted(groups)
        return {"people": {str(k): {"seconds": round(g[1], 1), "sentences": len(g[2]),
                                    "loudness_db": round(float(np.median(g[2])), 1) if g[2] else None}
                           for k, g in groups.items()},
                "similarity": {"%s-%s" % (a, b): round(float(groups[a][0] @ groups[b][0]), 2)
                               for i, a in enumerate(keys) for b in keys[i + 1:] if groups[a][1] and groups[b][1]}}

    def opinion(self):
        """{"people": [...], "same_person": [...]}, the shape Resolver.apply_model_opinion takes."""
        groups = {k: g for k, g in self.grouped().items() if g[1] >= MIN_VOICE}
        same = []
        keys = sorted(groups)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                similarity = float(groups[a][0] @ groups[b][0])
                confidence = max(0.0, min(1.0, (similarity - SAME_LOW) / (SAME_HIGH - SAME_LOW)))
                if confidence > 0:
                    same.append({"voices": [a, b], "confidence": confidence,
                                 "reason": "their voices sound alike (similarity %.2f)" % similarity})
        people = []
        loud = {k: float(np.median(g[2])) for k, g in groups.items() if len(g[2]) >= 2}
        if len(loud) >= 2 and self.me is None:          # with a baseline, the wearer is known by voice
            ranked = sorted(loud, key=loud.get, reverse=True)
            if loud[ranked[0]] - loud[ranked[1]] >= WEARER_DB:
                people.append({"voice": ranked[0], "name": None, "confidence": 0.0, "evidence": "",
                               "is_wearer": True})
        return {"people": people, "same_person": same}
