"""Aligns the speaker service's timeline (diart) with Deepgram's words, and turns it into reversible corrections.

Both use the band's audio clock. For each settled word:
  * one identified session speaker covers most of the word (>= SINGLE) and no overlap -> that speaker
  * the word falls in overlap (two voices, or an unidentified second voice)            -> ambiguous: unresolved,
    with candidates. Timestamp intersection alone cannot tell who said an overlapping word
  * no speech detected under the word, or the speaker is still "pending"               -> no opinion (left as is)
Session speakers become people of the conversation:
  * "wearer" (enrolled profile match)  -> You (person 0)
  * "other"                            -> the person whose clean words it shares, or a new person (enough speech)
  * "uncertain" (weak wearer match)    -> never a new person; its words become unresolved if they were on You
A later revision (e.g. a speaker becomes "wearer" after more clean speech) re-evaluates earlier words. Every change
is a proposal (engine "diart", with the service's revision and generation) applied reversibly: the original
lines are kept by the Resolver (revert_correction).
"""
import threading

from familiar.voices import ME

SINGLE = 0.6             # share of a word one speaker must cover
OVERLAP = 0.3            # share of a word in overlap that makes it ambiguous
MIN_ANCHORS = 3          # clean words a session speaker must share with a person to be mapped to them
ANCHOR_SHARE = 0.6
NEW_PERSON_SECONDS = 2.0


class DiartAttribution:
    def __init__(self, resolver, voices, apply=True, log=None):
        self.resolver, self.voices, self.apply = resolver, voices, apply
        self.log = log or (lambda text: None)
        self._lock = threading.RLock()
        self.reset()

    def reset(self):
        self.segments = []                # (generation, start, end, speakers tuple, overlap, unidentified)
        self.speakers = {}                # (generation, label) -> {"wearer", "clean_seconds", "similarity"}
        self.person = {}                  # (generation, label) -> person id (non-wearer)
        self.done_until = {}              # generation -> seconds covered by speaker events
        self.generation = None
        self.available = None
        self.epoch = self.resolver.epoch
        self.stats = {"events": 0, "proposals": 0, "applied": 0, "lag": []}

    # ------------------------------------------------------------------ events from the service

    def on_event(self, event):
        with self._lock:
            if self.resolver.epoch != self.epoch:
                self.reset()
            kind = event.get("type")
            gen = event.get("generation")
            self.stats["events"] += 1
            if kind == "status":
                self.available = event.get("attribution") == "available"
                self.resolver.diagnostics["speaker_service"] = {"attribution": event.get("attribution"),
                                                                "reason": event.get("reason")}
                self.resolver.set_attribution(event.get("attribution"))
                return
            if kind in ("ready", "generation"):
                self.generation = gen
                return
            if kind == "revision":
                key = (gen, event["speaker"])
                self.speakers.setdefault(key, {})
                self.speakers[key].update(wearer=event["wearer"], clean_seconds=event.get("clean_seconds", 0.0),
                                          similarity=event.get("similarity"))
                self.attribute(gen, event.get("applies_from") or 0.0, self.done_until.get(gen, 0.0),
                               revision=event.get("revision"))
                return
            if kind == "speaker":
                self.generation = gen
                for seg in event["segments"]:
                    self.segments.append((gen, seg["start"], seg["end"], tuple(seg["speakers"]), seg["overlap"],
                                          seg.get("unidentified", False)))
                self.segments = self.segments[-4000:]
                self.done_until[gen] = max(self.done_until.get(gen, 0.0), event["end"])
                if event.get("processing_lag") is not None:
                    self.stats["lag"] = (self.stats["lag"] + [event["processing_lag"]])[-200:]
                if self.available is None:
                    self.available = event.get("attribution") == "available"
                    self.resolver.set_attribution(event.get("attribution"))
                # words that just became fully covered (allow a little for words straddling the region edge)
                self.attribute(gen, event["start"] - 3.0, event["end"], revision=event.get("revision"))

    # ------------------------------------------------------------------ words

    def word_label(self, gen, start, end):
        """("single", label) | ("ambiguous", [labels]) | (None, None) for one word."""
        length = max(end - start, 1e-3)
        cover, overlap = {}, 0.0
        for g, a, b, speakers, ov, unid in self.segments:
            if g != gen or b <= start or a >= end:
                continue
            part = min(b, end) - max(a, start)
            if ov:
                overlap += part
            for s in speakers:
                cover[s] = cover.get(s, 0.0) + part
        if overlap / length >= OVERLAP:
            return "ambiguous", sorted(cover)
        if cover:
            best = max(cover, key=cover.get)
            if cover[best] / length >= SINGLE and len([s for s, c in cover.items() if c / length >= 0.2]) == 1:
                return "single", best
        return None, None

    def target_for(self, gen, kind, label, current):
        """The person a word should be on, None for unresolved, or "keep" for no opinion."""
        if kind == "ambiguous":
            return None
        if kind != "single":
            return "keep"
        info = self.speakers.get((gen, label), {})
        status = info.get("wearer")
        if status == "wearer":
            return ME
        if status in (None, "pending", "unknown"):
            return "keep" if status != "unknown" else self.person.get((gen, label), "keep")
        if status == "uncertain":                     # a weak wearer match: never a new person, never on You
            return None if current == ME else "keep"
        return self.person.get((gen, label), "keep")  # "other": mapped below

    def _person_of(self, gen, label):
        if self.speakers.get((gen, label), {}).get("wearer") == "wearer":
            return ME
        return self.person.get((gen, label))

    def _map_others(self, gen, words):
        """Session speakers that are not the wearer -> people, from words both agree on (anchors)."""
        votes = {}
        for w, kind, label in words:
            if kind != "single" or w["speaker"] in (None, ME) or w["status"] not in ("provider", "voice"):
                continue
            votes.setdefault(label, {}).setdefault(w["speaker"], 0)
            votes[label][w["speaker"]] += 1
        for (g, label), info in self.speakers.items():
            if g != gen or info.get("wearer") not in ("other", "unknown") or (g, label) in self.person:
                continue
            people = votes.get(label, {})
            used = {p for (gg, _), p in self.person.items() if gg == gen}
            if people:
                p, n = max(people.items(), key=lambda kv: kv[1])
                if n >= MIN_ANCHORS and n >= ANCHOR_SHARE * sum(people.values()) and p not in used:
                    self.person[(g, label)] = p
                    continue
            if info.get("wearer") == "other" and info.get("clean_seconds", 0.0) >= NEW_PERSON_SECONDS:
                new = self.voices._new_id() if self.voices is not None else max([0] + list(self.resolver.people)) + 1
                with self.resolver._lock:
                    self.resolver._person(new)
                self.person[(g, label)] = new

    def attribute(self, gen, start, end, revision=None):
        if end <= start:
            return []
        log = self.resolver.snapshot()["log"]
        words = []
        for line in log:
            for i, w in enumerate(line.get("words") or []):
                if start <= w[1] and w[2] <= end + 0.05:
                    kind, label = self.word_label(gen, w[1], w[2])
                    words.append(({"line": line["id"], "index": i, "start": w[1], "end": w[2], "text": w[0],
                                   "speaker": line["speaker"], "status": line.get("status")}, kind, label))
        if not words:
            return []
        self._map_others(gen, words)
        props, run = [], None
        for w, kind, label in words:
            target = self.target_for(gen, kind, label, w["speaker"])
            if target == "keep" or target == w["speaker"]:
                target = "keep"
            if target != "keep" and run and run["line"] == w["line"] and run["to"] == target and run["words"][1] == w["index"]:
                run["words"][1] = w["index"] + 1
                run["end"] = w["end"]
                continue
            if run:
                props.append(run)
                run = None
            if target != "keep":
                cands = [self._person_of(gen, l) for l in label] if kind == "ambiguous" else []
                cands = [c for c in cands if c is not None]
                run = {"engine": "diart", "line": w["line"], "words": [w["index"], w["index"] + 1], "start": w["start"],
                       "end": w["end"], "from": w["speaker"], "to": target, "confidence": None, "generation": gen,
                       "service_revision": revision,
                       "reason": "speaker service: %s" % ("overlapping speech" if kind == "ambiguous" else
                                                          "session speaker %s" % label),
                       "candidates": cands}
        if run:
            props.append(run)
        if not props:
            return []
        if any(p["to"] == ME for p in props) or any(ME in p["candidates"] for p in props):
            with self.resolver._lock:
                self.resolver._person(ME)                  # "You" exists before words are moved to it
        self.resolver.add_proposals(props)
        self.stats["proposals"] += len(props)
        if self.apply:
            keys = {(p["line"], tuple(p["words"]), p["to"]) for p in props}
            ids = [p["id"] for p in self.resolver.proposals
                   if p["engine"] == "diart" and not p["applied"] and (p["line"], tuple(p["words"]), p["to"]) in keys]
            done = self.resolver.apply_proposals(ids)
            self.stats["applied"] += len(done)
            if any(p["to"] == ME for p in props) and not self.resolver.people.get(ME, {}).get("is_me"):
                self.resolver.set_me(ME)
        return props
