"""Who is talking RIGHT NOW, from the sound alone (no transcript, no Deepgram, no names).

Every HOP seconds the tracker takes the last WINDOW seconds of the band's sound and, if there is speech in it,
a voiceprint (SpeechBrain ECAPA, on this laptop). It compares that with:
  * the wearer's enrolled baseline (familiar/enroll.py) -> "you"
  * voices it has learned during this conversation    -> "p1", "p2", ...
Scores are smoothed over the last few hops and a change of speaker needs CONFIRM hops in a row, so the answer
does not flicker on one noisy window. A voice that matches nobody for long enough becomes a new person.

Nothing is saved: the sound (last few seconds) and the session voices live in memory only.
"""
import threading
import time
from collections import deque

import numpy as np

SR = 16000
# Chosen on the synthetic scenarios (eval/tracker_eval.py --grid): 1.0 s / 0.25 s / smooth 2 / confirm 1 gave
# 72% of moments right, 5% wrong, ~0.9 s reaction; 1.5 s windows react ~2 s late. Real voices: --turn-test.
WINDOW = 1.0          # seconds of sound per decision
HOP = 0.25            # seconds between decisions
CONFIRM = 1           # decisions in a row before the speaker changes
SMOOTH = 2            # decisions whose scores are averaged
SPEECH_SHARE = 0.35   # share of a window's frames that must be speech
GATE_DB = 8.0         # dB above the running noise floor that counts as speech
ME_MARGIN = 0.04      # without a measured tracker threshold: baseline threshold - this (short windows score lower)
OTHER_SAME = 0.40     # a known session voice if this similar
OTHER_NEW = 0.25      # nobody known if below this for every voice
NEW_AFTER = 3         # hops of "nobody known" (with consistent windows) before a new person is started
LEARN_MARGIN = 0.10   # only windows this clearly decided teach a session voice ...
LEARN_MIN = 0.55      # ... that match it at least this well ...
STEADY = 0.80         # ... and sound like the window before (not a hand-over between two voices)
NEW_AFTER_WEAK = 9    # hops of "not clearly anyone" (consistent windows) before a new person anyway (~2 s)


class VoiceTracker:
    def __init__(self, encode, me=None, on_change=None, clock=time.monotonic, window=None, hop=None, smooth=None,
                 confirm=None):
        self.encode = encode
        self.window_s = window or WINDOW
        self.hop = hop or HOP
        self.smooth_n = smooth or SMOOTH
        self.confirm = confirm or CONFIRM
        self.me = None
        if me is not None:
            v = np.asarray(me["vector"], dtype=np.float32)
            # the turn test measures a threshold for these short windows; until then, derive one from the baseline
            t = me.get("tracker_threshold")
            self.me = {"vector": v / np.linalg.norm(v),
                       "threshold": float(t) if t is not None else float(me["threshold"]) - ME_MARGIN}
        self.on_change = on_change
        self.clock = clock
        self._buf = np.zeros(int(4 * SR), dtype=np.float32)
        self._written = 0
        self._lock = threading.Lock()
        self._floor = deque(maxlen=int(10 / 0.03))          # recent frame levels, for the noise floor
        self.reset()

    def reset(self):
        self.others = {}                                     # "p1" -> {"sum": vector, "n": windows}
        self.current, self.pending, self.pending_n = "silence", None, 0
        self.unknown = []                                    # recent unmatched voiceprints
        self.scores = deque(maxlen=self.smooth_n)            # recent {speaker: similarity}
        self.history = []                                    # (time, speaker, score) at each change
        self.last = {"speaker": "silence", "score": None, "level": -120.0, "scores": {}}
        self._count = 0
        self._raw = {}
        self._prev = None

    # ------------------------------------------------------------------ sound in

    def feed(self, pcm):
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768 if isinstance(pcm, (bytes, bytearray)) else pcm
        with self._lock:
            n, size = len(x), len(self._buf)
            if n >= size:
                self._buf[:] = x[-size:]
            else:
                self._buf = np.roll(self._buf, -n)
                self._buf[-n:] = x
            self._written += n

    def window(self, seconds=None):
        seconds = seconds or self.window_s
        with self._lock:
            if self._written < seconds * SR:
                return None
            return self._buf[-int(seconds * SR):].copy()

    # ------------------------------------------------------------------ one decision

    def _speech_share(self, x):
        n = int(0.03 * SR)
        frames = x[:len(x) // n * n].reshape(-1, n)
        db = 20 * np.log10(np.maximum(np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)), 1e-6))
        self._floor.extend(db.tolist())
        floor = float(np.percentile(self._floor, 10)) if self._floor else -90.0
        gate = max(floor + GATE_DB, -55.0)
        return float(np.mean(db >= gate)), float(np.median(db))

    def _centroid(self, key):
        o = self.others[key]
        return o["sum"] / np.linalg.norm(o["sum"])

    def step(self, x=None):
        """Decide on the latest window. Returns the current state dict."""
        x = self.window() if x is None else x
        if x is None:
            return self.last
        share, level = self._speech_share(x)
        self._raw = {}
        if share < SPEECH_SHARE:
            return self._settle("silence", None, {}, level)
        v = self.encode(x)
        sims = {}
        if self.me is not None:
            sims["you"] = float(v @ self.me["vector"])
        for key in self.others:
            sims[key] = float(v @ self._centroid(key))
        self._raw = {k: round(v_, 3) for k, v_ in sims.items()}       # this window alone (the turn test calibrates on it)
        self.scores.append(sims)
        smooth = {k: float(np.mean([s[k] for s in self.scores if k in s])) for k in sims}
        guess, score, created = self._guess(smooth, sims, v)
        steady = self._prev is not None and float(v @ self._prev) >= STEADY
        self._prev = v
        if guess in self.others and not created and steady:
            rivals = [s for k, s in sims.items() if k != guess]
            if sims[guess] >= LEARN_MIN and sims[guess] - max(rivals or [-1.0]) >= LEARN_MARGIN:
                self.others[guess]["sum"] = self.others[guess]["sum"] + v       # clear, single-voice windows only
                self.others[guess]["n"] += 1
        return self._settle(guess, score, smooth, level, created)

    def _guess(self, smooth, sims, v):
        """(speaker, score, created): "you", a session voice, a NEW session voice, or "unclear"."""
        if self.me is not None and smooth.get("you", -1) >= self.me["threshold"]:
            self.unknown = []
            return "you", smooth["you"], False
        others = {k: s for k, s in smooth.items() if k != "you"}
        best = max(others, key=others.get) if others else None
        if best is not None and others[best] >= OTHER_SAME:
            self.unknown = []
            return best, others[best], False
        not_me = self.me is None or sims.get("you", -1) < self.me["threshold"] - 0.1
        strong = not_me and all(s < OTHER_NEW for k, s in sims.items() if k != "you")
        weak = not_me and all(s < OTHER_SAME - 0.05 for k, s in sims.items() if k != "you")
        if not weak:
            self.unknown = []
            return "unclear", None, False
        self.unknown.append((v, strong))
        need = NEW_AFTER if all(st for _, st in self.unknown[-NEW_AFTER:]) else NEW_AFTER_WEAK
        recent = [u for u, _ in self.unknown[-need:]]
        if len(recent) >= need and min(float(a @ b) for a in recent for b in recent) >= OTHER_SAME - 0.05:
            self._count += 1
            key = "p%d" % self._count
            self.others[key] = {"sum": np.sum(recent, axis=0), "n": len(recent)}
            self.unknown = []
            self.scores.clear()
            return key, float(np.mean([float(a @ self._centroid(key)) for a in recent])), True
        return "unclear", None, False

    def _settle(self, guess, score, smooth, level, created=False):
        if guess == self.current:
            self.pending, self.pending_n = None, 0
        elif created:                                        # a new person already took NEW_AFTER windows
            self._change(guess, score)
        elif guess == self.pending:
            self.pending_n += 1
            if self.pending_n >= self.confirm:
                self._change(guess, score)
        else:
            self.pending, self.pending_n = guess, 1
            if self.confirm <= 1:
                self._change(guess, score)
        self.last = {"speaker": self.current, "score": None if score is None else round(score, 3), "level": round(level, 1),
                     "scores": {k: round(s, 3) for k, s in smooth.items()}, "raw": dict(self._raw),
                     "people": sorted(self.others), "time": round(self.clock(), 2)}
        return self.last

    def _change(self, guess, score):
        self.current, self.pending, self.pending_n = guess, None, 0
        self.history.append((round(self.clock(), 2), guess, None if score is None else round(score, 3)))
        if self.on_change:
            self.on_change(guess, score)

    # ------------------------------------------------------------------ running live

    def run(self, stop):
        """Decide every HOP seconds until `stop` (a threading.Event) is set."""
        while not stop.wait(self.hop):
            try:
                self.step()
            except Exception as problem:                     # one bad window must not stop the tracker
                print("Voice tracker: %s" % type(problem).__name__)
