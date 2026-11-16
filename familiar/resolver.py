"""Turns a diarized conversation into "who is who".

The speech service tells us WHEN someone different is talking ("voice 0", "voice 1", ...)
but not WHO they are. This module listens to what is said and picks up name clues:

  * self introductions        "Hi, I'm Maya"  /  "my name is Maya Chen"  /  "call me Dev"
  * an answer to a question   "What's your name?"  ->  "Priya."
  * being addressed           "Nice to meet you, Dev"  ->  names the person who spoke just before
  * small facts about them    "I work at Northline", "I'm a product designer"

Anything it is not sure about is shown as "possible" so the person using the app can
confirm, fix or fill in the name by hand. A name typed by the user always wins.

A language model can add a second opinion on top (apply_model_opinion, see familiar/gemini.py).
Every name keeps its source: "regex" (the rules here), "gemini" or "typed".

It only knows people inside the current conversation. Nothing is stored on disk.
No dependencies, Python 3.9+.
"""
import re
import threading
import time

NAME_TOKEN = r"[A-Z][A-Za-z'-]{1,19}"
NAME = r"%s(?:\s+%s)?" % (NAME_TOKEN, NAME_TOKEN)

# Words that look like names when they start a sentence but are not.
STOP = set("""
i i'm i've i'll i'd im a an the and or but so not no yes yeah yep nope ok okay oh um uh hm hmm well
good fine great doing here there just sorry sure glad happy excited trying looking going working
really very still now back new interested curious tired late early actually definitely honestly
hungry thinking wondering about from in at on with also one two kinda kind maybe probably totally
thanks thank please hello hi hey welcome everyone everybody guys all again today tonight tomorrow
yesterday monday tuesday wednesday thursday friday saturday sunday january february march april
may june july august september october november december it's it that's this that what who where
when why how which student engineer designer developer founder here's
""".split())

GREETING = r"(?i:hi|hey|hello|hi there|hey there)"

# "I am X" -> the speaker's own name.
SELF_PATTERNS = [
    re.compile(r"(?i:\bmy name(?: is|'s)|\bthe name(?: is|'s)|\bname's)\s+(?P<n>%s)" % NAME),
    re.compile(r"(?i:\bi am|\bi'm|\bim)\s+(?P<n>%s)" % NAME),
    re.compile(r"(?i:\bcall me|\bthey call me|\beveryone calls me|\byou can call me)\s+(?P<n>%s)" % NAME),
    re.compile(r"^(?:%s[,.!]?\s+)?(?i:this is)\s+(?P<n>%s)" % (GREETING, NAME)),
    re.compile(r"^%s[,.!]?\s+(?i:it's|it is)\s+(?P<n>%s)" % (GREETING, NAME)),
]
# Speech services sometimes leave an unusual name in lower case; accept it after a strong phrase only.
SELF_LOWER = re.compile(r"(?i:\bmy name(?: is|'s)|\bcall me)\s+(?P<n>[a-z][a-z'-]{1,19})\b")

# "... Dev" said TO someone -> the name belongs to the person who spoke just before.
ADDRESS_PATTERNS = [
    re.compile(r"(?i:(?:nice|good|great|lovely|pleasure|pleased|happy) (?:to meet|meeting|to see) you)[,.!]?\s+(?P<n>%s)" % NAME),
    re.compile(r"(?i:\b(?:thanks|thank you|cheers|appreciate it|sounds good|got it|okay|ok|sure|yeah|right)),\s+(?P<n>%s)" % NAME),
    re.compile(r"(?:^|[.!?,]\s+)(?i:hi|hey|hello|welcome|hey there|hi there),?\s+(?P<n>%s)" % NAME),
]

ASK_NAME = re.compile(r"(?i)\b(what'?s your name|what is your name|your name\b|who are you|and your name)")
ANSWER = re.compile(r"^(?:(?i:it'?s|i'?m|i am|that'?s|they call me|name'?s)\s+)?(?P<n>%s)[.!?]?$" % NAME)

ORG_WORD = r"[A-Z][\w&-]*(?:\.[\w&-]+)*"
ORG = r"%s(?:\s+%s){0,2}" % (ORG_WORD, ORG_WORD)
CLUE_PATTERNS = [
    (re.compile(r"(?i:\bi(?:'m| am) (?:a|an))\s+(?P<r>[a-z][a-z -]{2,38}?)\s+(?i:at|for)\s+(?P<o>%s)" % ORG),
     lambda m: ["%s" % m.group("r").strip().capitalize(), "Works at %s" % m.group("o")]),
    (re.compile(r"(?i:\bi (?:work|worked) (?:at|for))\s+(?P<o>%s)" % ORG),
     lambda m: ["Works at %s" % m.group("o")]),
    (re.compile(r"(?i:\bi(?:'m| am) (?:a|an))\s+(?P<r>[a-z][a-z -]{2,38}?)(?=\s+(?i:and|who|working|building|from)\b|[.,!?]|$)"),
     lambda m: ["%s" % m.group("r").strip().capitalize()]),
    (re.compile(r"(?i:\bi(?:'m| am) from|\bi live in|\bi(?:'m| am) based in|\bi(?:'m| am) living in)\s+(?P<p>%s)" % ORG),
     lambda m: ["From %s" % m.group("p")]),
    (re.compile(r"(?i:\bi(?:'m| am)? (?:study|studying))\s+(?P<s>[a-z][a-z ]{2,28}?)(?=\s+(?i:at|in)\b|[.,!?]|$)"),
     lambda m: ["Studies %s" % m.group("s").strip()]),
    (re.compile(r"(?i:\bi(?:'m| am)? (?:building|working on|work on|making))\s+(?P<b>[^.,!?]{3,48})"),
     lambda m: ["Building %s" % re.split(r"\s+(?:and|but|so)\s+", m.group("b").strip())[0]]),
]

STATUS_RANK = {"unknown": 0, "possible": 1, "heard": 2, "confirmed": 3}

# Speaker attribution of a settled line, kept separate from whether its TEXT is final:
#   "provider"   Deepgram's speaker label, mapped to this conversation's people
#   "voice"      decided by voiceprints (a re-identified label, or part of a sentence split where the voice changed)
#   "corrected"  moved by a second pass on recorded audio (reversible)
#   "uncertain"  who said it could not be resolved: the line has no speaker, maybe some candidates
OVERLAP_MIN = 0.15        # seconds: two speakers' WORDS must overlap at least this much to count as overlap

# Second opinion from a language model (familiar/gemini.py)
MODEL_HEARD = 0.8         # at or above: "heard"
MODEL_POSSIBLE = 0.5      # at or above: "possible"; below: ignored
MODEL_AUTO_MERGE = 0.85   # same-person suggestion this sure, and the voices never spoke at once: join them


def _norm(text):
    return (text or "").replace("\u2019", "'").replace("\u2018", "'").strip()


def clean_name(raw):
    """"Maya Chen" stays, "Maya And" becomes "Maya", "Good" becomes None."""
    if not raw:
        return None
    tokens = [t.strip("'-") for t in raw.split()]
    tokens = [t for t in tokens if t]
    while tokens and tokens[-1].lower() in STOP:
        tokens.pop()
    if not tokens or tokens[0].lower() in STOP:
        return None
    name = " ".join(t[:1].upper() + t[1:] for t in tokens[:2])
    if name.lower().endswith("'s"):
        name = name[:-2]
    return name if len(name) >= 2 else None


def find_self_name(text):
    """The speaker's own name in this sentence, or None."""
    text = _norm(text)
    for pattern in SELF_PATTERNS:
        for m in pattern.finditer(text):
            name = clean_name(m.group("n"))
            if name:
                return name
    m = SELF_LOWER.search(text)
    if m:
        return clean_name(m.group("n"))
    return None


def find_addressed_name(text):
    """A name said TO someone ("Nice to meet you, Dev"), or None."""
    text = _norm(text)
    for pattern in ADDRESS_PATTERNS:
        m = pattern.search(text)
        if m:
            name = clean_name(m.group("n"))
            if name:
                return name
    return None


def _squash(text):
    return re.sub(r"\s+", " ", _norm(text)).lower()


def _as_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _as_conf(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if value != value else max(0.0, min(1.0, value))


def find_answer_name(text):
    m = ANSWER.match(_norm(text))
    return clean_name(m.group("n")) if m else None


def extract_clues(text):
    text = _norm(text)
    found = []
    for pattern, make in CLUE_PATTERNS:
        for m in pattern.finditer(text):
            for clue in make(m):
                if clue and clue.lower() not in [c.lower() for c in found]:
                    found.append(clue)
    return found[:3]


class Resolver:
    """Feed it diarized lines; ask it for a snapshot to draw."""

    def __init__(self, on_change=None, on_event=None, clock=time.time, window=45.0):
        self._lock = threading.RLock()
        self._clock = clock
        self._window = window
        self.on_change = on_change
        self.on_event = on_event
        self.on_final = None      # called (no arguments) after each settled line: the model advisor's nudge
        self.assist = "off"       # the second opinion: "off", "on" or "rules" (unavailable: rules only)
        self.attribution = None   # the speaker service: None (not used), "available" or "unavailable"
        self.epoch = 0
        self.diagnostics = {}     # numbers for /debug (voice matching), never shown on the page
        self.reset(notify=False)

    # ------------------------------------------------------------------ state

    def reset(self, notify=True):
        with self._lock:
            self.people = {}          # speaker id -> dict
            self.order = []           # speaker ids by first appearance
            self.alias = {}           # merged id -> kept id
            self.history = []         # (speaker, time) of recent final lines
            self.log = []             # transcript lines
            self.events = []          # name pick-ups, for the toast
            self.now = None
            self.interim = None
            self._event_id = 0
            self._line_id = 0
            self.rev = 0
            self.epoch += 1           # a model reply about an older conversation is dropped
            self.spans = []           # (speaker, start, end) of recent final lines, to tell overlapping voices
            self._open = {}           # speaker -> time their current line started
            self.suggestions = []     # "Same person?" questions for the page
            self.proposals = []       # second-pass / review suggestions about who said which words
            self.corrections = []     # applied ones, each with the original lines (reversible)
            self._proposal_id = 0
            self.declined = set()     # frozenset({a, b}) the user said are different people
            self._suggestion_id = 0
            self.me_locked = False    # someone (the user or the demo) said who the wearer is
            self.notes = None         # Gemini's grounded notes: topics, mentioned names, follow-ups
        if notify:
            self._changed()

    def _person(self, sid):
        sid = self.alias.get(sid, sid)
        if sid not in self.people:
            self.order.append(sid)
            self.people[sid] = {
                "id": sid, "label": "Voice " + chr(65 + (len(self.order) - 1) % 26),
                "name": None, "status": "unknown", "evidence": None, "alts": [],
                "rejected": [], "is_me": False, "clues": [], "lines": 0, "last_text": "", "source": None,
            }
        return self.people[sid]

    def _changed(self):
        with self._lock:
            self.rev += 1
            snap = self.snapshot()
        if self.on_change:
            self.on_change(snap)

    def _event(self, kind, person, why):
        self._event_id += 1
        event = {"id": self._event_id, "kind": kind, "speaker": person["id"],
                 "name": person["name"], "status": person["status"], "why": why,
                 "source": person["source"], "display": person["name"] or person["label"]}
        self.events = (self.events + [event])[-12:]
        if self.on_event:
            self.on_event(event)

    # ------------------------------------------------------------------ input

    def feed(self, speaker, text, final=True, t=None, meta=None):
        """One piece of speech. speaker: the voice number (None = same as before, or, with meta, unresolved).

        meta (live mode) describes a settled line: {"start", "end"} in audio seconds, "words"
        [(word, start, end), ...], "status" (see above), "candidates" [person ids] when uncertain,
        "label" (the provider's speaker label, scoped to its connection)."""
        text = _norm(text)
        if not text:
            return
        now = self._clock() if t is None else t
        with self._lock:
            if meta is not None and final:
                self.interim = None
                if speaker is None:                     # unresolved: kept in the transcript, nobody's words
                    self._line(None, text, meta)
                else:
                    person = self._person(self.alias.get(speaker, speaker))
                    self.now = person["id"]
                    if meta.get("start") is not None and meta.get("end") is not None:
                        self.spans = (self.spans + [(person["id"], meta["start"], meta["end"])])[-200:]
                    self._final(person, text, now, meta)
            else:
                if speaker is None:
                    speaker = self.now if self.now is not None else 0
                speaker = self.alias.get(speaker, speaker)
                person = self._person(speaker)
                self.now = person["id"]
                self._open.setdefault(person["id"], now)
                if not final:
                    self.interim = [{"speaker": person["id"], "text": text}]
                else:
                    self.interim = None
                    self.spans = (self.spans + [(person["id"], self._open.pop(person["id"]), now)])[-200:]
                    self._final(person, text, now)
        self._changed()
        if final and self.on_final:
            self.on_final()

    def set_interim(self, pieces):
        """The provisional text, REPLACING the previous one: [{"speaker": person id or None, "text"}].
        It never creates a person: a voice that is only guessed at gets no card until a line settles."""
        with self._lock:
            pieces = [{"speaker": self.alias.get(p.get("speaker"), p.get("speaker")), "text": _norm(p.get("text"))}
                      for p in pieces or [] if _norm(p.get("text"))]
            self.interim = pieces or None
            if pieces and pieces[-1]["speaker"] in self.people:
                self.now = pieces[-1]["speaker"]
        self._changed()

    def _line(self, sid, text, meta=None):
        """Append a settled line to the transcript, with its timing, words, attribution and overlaps."""
        self._line_id += 1
        line = {"id": self._line_id, "speaker": sid, "text": text}
        if meta is not None:
            line.update({"start": meta.get("start"), "end": meta.get("end"),
                         "words": [list(w) for w in meta.get("words") or []],
                         "status": meta.get("status") or ("uncertain" if sid is None else "provider"),
                         "candidates": [c for c in meta.get("candidates") or [] if c in self.people],
                         "label": meta.get("label"), "overlaps": []})
            self._mark_overlaps(line)
        self.log = (self.log + [line])[-60:]
        return line

    def _mark_overlaps(self, line):
        """Overlap only from evidence: two different speakers' WORDS overlapping in audio time by OVERLAP_MIN."""
        if line["speaker"] is None or not line.get("words"):
            return
        for other in self.log[-10:]:
            if other.get("speaker") in (None, line["speaker"]) or not other.get("words"):
                continue
            spans = [(max(a[1], b[1]), min(a[2], b[2])) for a in line["words"] for b in other["words"]]
            spans = [(a, b) for a, b in spans if b - a > 0.0]
            if spans and sum(b - a for a, b in spans) >= OVERLAP_MIN:
                start, end = round(min(a for a, _ in spans), 2), round(max(b for _, b in spans), 2)
                line["overlaps"].append({"with": other["speaker"], "line": other["id"], "start": start, "end": end})
                other.setdefault("overlaps", []).append({"with": line["speaker"], "line": line["id"],
                                                         "start": start, "end": end})

    def _final(self, person, text, now, meta=None):
        sid = person["id"]
        previous = self.history[-1] if self.history else None
        prev_text = self.log[-1]["text"] if self.log else ""
        prev_other = previous[0] if previous and previous[0] != sid else None

        # 1. name clues inside this sentence
        self_name = find_self_name(text)
        why = None
        if self_name:
            why = "said “%s”" % text
        elif prev_other is not None and ASK_NAME.search(prev_text):
            self_name = find_answer_name(text)
            if self_name:
                why = "answered “%s” when asked their name" % text
        if self_name:
            self._offer(person, self_name, "heard", why, "regex")

        addressed = find_addressed_name(text)
        if addressed:
            target = self._recent_other(sid, now)
            if target is not None and (not self_name or addressed != self_name):
                target_person = self._person(target)
                if addressed != person["name"]:
                    self._offer(target_person, addressed, "possible", "was called “%s” by %s" % (
                        addressed, "you" if person["is_me"] else (person["name"] or person["label"])), "regex")

        # 2. small facts the speaker says about themselves
        for clue in extract_clues(text):
            if clue.lower() not in [c.lower() for c in person["clues"]]:
                person["clues"] = (person["clues"] + [clue])[-5:]

        person["lines"] += 1
        person["last_text"] = text
        self.history = (self.history + [(sid, now)])[-30:]
        self._line(sid, text, meta)

    def _recent_other(self, sid, now):
        for speaker, when in reversed(self.history):
            if now - when > self._window:
                break
            if speaker != sid:
                return speaker
        return None

    def _offer(self, person, name, status, why, source):
        """A name candidate. Typed names win; heard beats possible; first heard name sticks."""
        if name.lower() in [r.lower() for r in person["rejected"]]:
            return
        if person["status"] == "confirmed":
            if name != person["name"] and name not in person["alts"]:
                person["alts"].append(name)
            return
        if person["name"] and person["name"] == name:
            if STATUS_RANK[status] > STATUS_RANK[person["status"]]:
                person["status"], person["evidence"], person["source"] = status, why, source
            return
        if STATUS_RANK[status] > STATUS_RANK[person["status"]] or not person["name"]:
            if person["name"] and person["name"] not in person["alts"]:
                person["alts"].append(person["name"])
            person["name"], person["status"], person["evidence"], person["source"] = name, status, why, source
            self._event("name", person, why)
        elif name not in person["alts"]:
            person["alts"].append(name)

    # ------------------------------------------------------------------ the person at the keyboard

    def set_name(self, speaker, name):
        """The user typed a name. It is final until they change it."""
        name = clean_name(_norm(name).title()) if name and name.strip() else None
        with self._lock:
            person = self._person(speaker)
            if not name:
                return self.clear_name(speaker)
            if person["name"] and person["name"] != name and person["name"] not in person["alts"]:
                person["alts"].append(person["name"])
            person["name"], person["status"], person["evidence"] = name, "confirmed", "you typed it"
            person["source"] = "typed"
            if name in person["rejected"]:
                person["rejected"].remove(name)
        self._changed()

    def clear_name(self, speaker):
        """"Not Maya?": forget the name, and do not offer it for this voice again."""
        with self._lock:
            person = self._person(speaker)
            if person["name"] and person["name"] not in person["rejected"]:
                person["rejected"].append(person["name"])
            person["name"], person["status"], person["evidence"], person["source"] = None, "unknown", None, None
        self._changed()

    def set_me(self, speaker, flag=True):
        with self._lock:
            self.me_locked = True
            for p in self.people.values():
                p["is_me"] = False
            if flag:
                self._person(speaker)["is_me"] = True
        self._changed()

    def merge(self, src, dst):
        """Diarization sometimes splits one person into two voices: join them."""
        with self._lock:
            src, dst = self.alias.get(src, src), self.alias.get(dst, dst)
            if src == dst or src not in self.people or dst not in self.people:
                return
            a, b = self.people[src], self.people[dst]
            if a["name"] and STATUS_RANK[a["status"]] > STATUS_RANK[b["status"]]:
                if b["name"] and b["name"] != a["name"] and b["name"] not in b["alts"]:
                    b["alts"].append(b["name"])
                b["name"], b["status"], b["evidence"], b["source"] = a["name"], a["status"], a["evidence"], a["source"]
            elif a["name"] and a["name"] != b["name"] and a["name"] not in b["alts"]:
                b["alts"].append(a["name"])
            b["rejected"] = list(dict.fromkeys(b["rejected"] + a["rejected"]))
            for clue in a["clues"]:
                if clue not in b["clues"]:
                    b["clues"].append(clue)
            b["lines"] += a["lines"]
            b["is_me"] = b["is_me"] or a["is_me"]
            b["alts"] = list(dict.fromkeys(b["alts"] + a["alts"]))
            del self.people[src]
            self.order = [s for s in self.order if s != src]
            for key, value in list(self.alias.items()):
                if value == src:
                    self.alias[key] = dst
            self.alias[src] = dst
            self.history = [(dst if s == src else s, w) for s, w in self.history]
            for line in self.log:
                if line["speaker"] == src:
                    line["speaker"] = dst
            if self.now == src:
                self.now = dst
            self.spans = [(dst if s == src else s, a0, a1) for s, a0, a1 in self.spans]
            if src in self._open:
                start = self._open.pop(src)
                self._open[dst] = min(start, self._open.get(dst, start))
            self.declined = {frozenset(dst if v == src else v for v in pair) for pair in self.declined}
            kept = []
            for sug in self.suggestions:
                sug["voices"] = [dst if v == src else v for v in sug["voices"]]
                if sug["voices"][0] != sug["voices"][1]:
                    kept.append(sug)
            self.suggestions = kept
        self._changed()

    # ------------------------------------------------------------------ second opinion (familiar/gemini.py)

    def model_input(self, lines=20):
        """What the model sees: the last settled lines, who is who so far, and the names the user rejected."""
        with self._lock:
            if not self.log:
                return None
            people = [{"voice": sid, "name": self.people[sid]["name"], "status": self.people[sid]["status"],
                       "typed": self.people[sid]["status"] == "confirmed", "is_wearer": self.people[sid]["is_me"],
                       "alts": list(self.people[sid]["alts"])}
                      for sid in self.order]
            return {"epoch": self.epoch,
                    "lines": [{"voice": line["speaker"], "text": line["text"]} for line in self.log[-lines:]],
                    "people": people,
                    "rejected": {sid: list(self.people[sid]["rejected"]) for sid in self.order
                                 if self.people[sid]["rejected"]},
                    "declined": [sorted(pair) for pair in self.declined],
                    "wearer_known": self.me_locked}

    def set_attribution(self, state):
        with self._lock:
            if state == self.attribution:
                return
            self.attribution = state
        self._changed()

    def set_notes(self, notes):
        with self._lock:
            self.notes = notes
        self._changed()

    def set_assist(self, state):
        with self._lock:
            if state == self.assist:
                return
            self.assist = state
        self._changed()

    def overlapped(self, a, b):
        """True if voices a and b were ever heard speaking at the same time (then they are two people)."""
        with self._lock:
            a, b = self.alias.get(a, a), self.alias.get(b, b)
            mine = [(s, e) for v, s, e in self.spans if v == a]
            theirs = [(s, e) for v, s, e in self.spans if v == b]
            return any(s1 < e2 and s2 < e1 for s1, e1 in mine for s2, e2 in theirs)

    SOURCE_LABEL = {"gemini": "Gemini", "voice": "Voice match"}

    def apply_model_opinion(self, reply, epoch=None, source="gemini"):
        """Merge a model's reply into the regex result. Typed names always win; weak or unbacked names are ignored."""
        if not isinstance(reply, dict):
            return
        with self._lock:
            if epoch is not None and epoch != self.epoch:
                return                                          # the conversation was reset meanwhile
            heard = _squash(" \n ".join(line["text"] for line in self.log))
            people = reply.get("people") if isinstance(reply.get("people"), list) else []
            for item in people:
                if isinstance(item, dict):
                    self._model_name(item, heard, source)
            wearers = [_as_int(i.get("voice")) for i in people if isinstance(i, dict) and i.get("is_wearer") is True]
            wearers = [self.alias.get(v, v) for v in wearers if v is not None]
            if not self.me_locked and len(set(wearers)) == 1 and wearers[0] in self.people:
                self.people[wearers[0]]["is_me"] = True
                self.me_locked = True
            same = reply.get("same_person") if isinstance(reply.get("same_person"), list) else []
            for item in same:
                if isinstance(item, dict):
                    self._model_same(item, source)
        self._changed()

    def _model_name(self, item, heard, source="gemini"):
        sid = _as_int(item.get("voice"))
        if sid is None:
            return
        sid = self.alias.get(sid, sid)
        if sid not in self.people or not isinstance(item.get("name"), str):
            return
        name = clean_name(_norm(item["name"]))
        confidence = _as_conf(item.get("confidence"))
        if not name or confidence < MODEL_POSSIBLE:
            return
        evidence = _norm(item.get("evidence") if isinstance(item.get("evidence"), str) else "").strip(" \"“”")
        quote = _squash(evidence)
        # the quote must really be in the transcript, and must contain the name: no made-up evidence
        if not quote or quote not in heard or name.split()[0].lower() not in quote:
            return
        if not self.grounded(sid, name):                # a name merely MENTIONED is not anyone's identity
            return
        status = "heard" if confidence >= MODEL_HEARD else "possible"
        self._offer(self.people[sid], name, status, "%s: “%s”" % (self.SOURCE_LABEL.get(source, source), evidence),
                    source)

    FIRST_PERSON = re.compile(r"(?i)\b(i'?m|i am|my name|name'?s|call me|me as|it'?s|this is|known as)\b")

    def grounded(self, sid, name):
        """Is `name` tied to person `sid` by what was said, with attributed lines only? Either sid says it in a
        first-person way ("I'm X", "people know me as X", or just "X." after being asked), or another person
        says it TO sid (addressed, in the line right before or after one of sid's lines). A name that is only
        mentioned ("I talked to Maya yesterday") does not count, and unresolved lines never count."""
        first = name.split()[0].lower()
        has = lambda text: re.search(r"\b%s\b" % re.escape(first), text, re.I) is not None
        log = [l for l in self.log if l.get("speaker") is not None and l.get("status") != "uncertain"]
        for i, line in enumerate(log):
            who = self.alias.get(line["speaker"], line["speaker"])
            if not has(line["text"]):
                continue
            if who == sid:
                if find_self_name(line["text"]) == name or self.FIRST_PERSON.search(line["text"]):
                    return True
                if i > 0 and ASK_NAME.search(log[i - 1]["text"]) and find_answer_name(line["text"]):
                    return True
            else:
                addressed = find_addressed_name(line["text"])
                n = re.escape(first)
                vocative = re.search(r"(?i)(^|[,.!?]\s*|\b(?:hi|hey|hello|thanks|thank you|bye|welcome|sorry|cheers|"
                                     r"morning|yes|no|okay|ok)\s+)%s\s*([,.!?]|$)|,\s*%s\s*[.!?]?$" % (n, n), line["text"])
                near = [self.alias.get(l["speaker"], l["speaker"]) for l in log[max(0, i - 1):i + 2] if l is not line]
                if (addressed and addressed.split()[0].lower() == first or vocative) and sid in near:
                    return True
        return False

    def _model_same(self, item, source="gemini"):
        voices = item.get("voices")
        if not isinstance(voices, list) or len(voices) != 2:
            return
        a, b = (_as_int(v) for v in voices)
        if a is None or b is None:
            return
        a, b = self.alias.get(a, a), self.alias.get(b, b)
        confidence = _as_conf(item.get("confidence"))
        if a == b or a not in self.people or b not in self.people or confidence < MODEL_POSSIBLE:
            return
        pair = frozenset((a, b))
        if pair in self.declined:
            return
        pa, pb = self.people[a], self.people[b]
        clash = (pa["name"] and pb["name"] and pa["name"] != pb["name"]
                 and STATUS_RANK[pa["status"]] >= 2 and STATUS_RANK[pb["status"]] >= 2)
        reason = str(item.get("reason") or "")[:160]
        if confidence >= MODEL_AUTO_MERGE and not clash and not self.overlapped(a, b):
            keep, drop = self._merge_order(a, b)
            dropped = self.people[drop]["name"] or self.people[drop]["label"]
            self.merge(drop, keep)                              # RLock: fine inside the lock
            kept = self.people[keep]
            self._event("merge", kept, "%s joined %s into %s: %s" % (
                self.SOURCE_LABEL.get(source, source), dropped, kept["name"] or kept["label"], reason or "same person"))
            return
        if any(frozenset(s["voices"]) == pair for s in self.suggestions):
            return
        self._suggestion_id += 1
        self.suggestions = (self.suggestions + [{"id": self._suggestion_id, "voices": sorted((a, b)),
                                                 "confidence": round(confidence, 2), "reason": reason,
                                                 "source": source}])[-4:]

    def _merge_order(self, a, b):
        """(keep, drop): keep the voice with the surer name, then the one that spoke more, then the older one."""
        def rank(sid):
            p = self.people[sid]
            return (STATUS_RANK[p["status"]] if p["name"] else -1, p["lines"], -self.order.index(sid))
        return (a, b) if rank(a) >= rank(b) else (b, a)

    # ------------------------------------------------------------------ second-pass corrections (reversible)

    def add_proposals(self, proposals):
        """Suggestions from a second pass or a review: {"engine", "line", "words": [i, j] (end exclusive),
        "start", "end", "to" (person id or None = unresolved), "confidence", "reason"}. Kept, not applied."""
        with self._lock:
            known = {(p["line"], tuple(p["words"]), p["to"], p["engine"]): p for p in self.proposals}
            for prop in proposals:
                key = (prop["line"], tuple(prop["words"]), prop["to"], prop["engine"])
                if key in known:                        # the same suggestion again: keep the newest evidence
                    if not known[key]["applied"]:
                        known[key].update({k: v for k, v in prop.items() if k not in ("id", "applied")})
                    continue
                self._proposal_id += 1
                self.proposals = (self.proposals + [dict(prop, id=self._proposal_id, applied=False)])[-40:]
                known[key] = self.proposals[-1]
        self._changed()

    def apply_proposal(self, proposal_id):
        done = self.apply_proposals([proposal_id])
        return done[0] if done else None

    def apply_proposals(self, proposal_ids):
        """Move the proposals' words to their speakers. Each affected line is split into runs of one speaker;
        the original line is kept so revert_correction() can put it back."""
        done = []
        with self._lock:
            props = [p for p in self.proposals if p["id"] in set(proposal_ids) and not p["applied"]]
            for line_id in sorted({p["line"] for p in props}):
                index = next((i for i, l in enumerate(self.log) if l["id"] == line_id), None)
                if index is None or not self.log[index].get("words"):
                    continue
                line = self.log[index]
                words = line["words"]
                who = [(line["speaker"], line.get("status"))] * len(words)
                mine = [p for p in props if p["line"] == line_id and 0 <= p["words"][0] < p["words"][1] <= len(words)]
                if not mine:
                    continue
                cands = {}
                for p in mine:
                    to = p["to"] if p["to"] in self.people else None
                    for i in range(*p["words"]):
                        who[i] = (to, "corrected" if to is not None else "uncertain")
                        if to is None and p.get("candidates"):
                            cands[i] = [c for c in p["candidates"] if c in self.people]
                new_lines, i = [], 0
                while i < len(words):
                    j = i
                    while j < len(words) and who[j] == who[i]:
                        j += 1
                    speaker, status = who[i]
                    self._line_id += 1
                    new_lines.append({"id": self._line_id, "speaker": speaker, "text": " ".join(w[0] for w in words[i:j]),
                                      "start": words[i][1], "end": words[j - 1][2], "words": words[i:j],
                                      "status": status,
                                      "candidates": (cands.get(i) or [line["speaker"]]) if speaker is None else [],
                                      "label": line.get("label"), "overlaps": []})
                    i = j
                self.log[index:index + 1] = new_lines
                old = self.people.get(line["speaker"])
                for nl in new_lines:
                    if nl["speaker"] == line["speaker"]:
                        continue
                    if old and old["status"] != "confirmed" and old["evidence"] and nl["text"] in old["evidence"]:
                        if old["name"] and old["name"] not in old["alts"]:
                            old["alts"].append(old["name"])
                        old["name"], old["status"], old["evidence"], old["source"] = None, "unknown", None, None
                    if nl["speaker"] is not None:
                        found = find_self_name(nl["text"])
                        if found:
                            self._offer(self.people[nl["speaker"]], found, "heard", "said “%s”" % nl["text"], "regex")
                for p in mine:
                    p["applied"] = True
                correction = {"id": mine[0]["id"], "proposals": [p["id"] for p in mine], "line": line_id,
                              "to": sorted({str(p["to"]) for p in mine}), "engine": mine[0]["engine"],
                              "original": dict(line), "new_ids": [l["id"] for l in new_lines]}
                self.corrections.append(correction)
                done.append(correction)
        self._changed()
        return done

    def dismiss_proposal(self, proposal_id):
        with self._lock:
            self.proposals = [p for p in self.proposals if p["id"] != proposal_id or p["applied"]]
        self._changed()

    def revert_correction(self, correction_id):
        """Put the original line back (names picked up by the correction are not undone)."""
        with self._lock:
            c = next((c for c in self.corrections if c["id"] == correction_id and not c.get("reverted")), None)
            if c is None:
                return False
            ids = set(c["new_ids"])
            first = next((i for i, l in enumerate(self.log) if l["id"] in ids), None)
            self.log = [l for l in self.log if l["id"] not in ids]
            self.log.insert(first if first is not None else len(self.log), c["original"])
            c["reverted"] = True
            for prop in self.proposals:
                if prop["id"] in c.get("proposals", [c["id"]]):
                    prop["applied"] = False
        self._changed()
        return True

    def answer_suggestion(self, suggestion_id, same):
        """"Same person?" answered on the page."""
        with self._lock:
            sug = next((s for s in self.suggestions if s["id"] == suggestion_id), None)
            if not sug:
                return
            self.suggestions = [s for s in self.suggestions if s is not sug]
            a, b = sug["voices"]
            if same and a in self.people and b in self.people:
                keep, drop = self._merge_order(a, b)
                self.merge(drop, keep)
                return
            self.declined.add(frozenset((a, b)))
        self._changed()

    # ------------------------------------------------------------------ output

    def snapshot(self):
        with self._lock:
            people = []
            for index, sid in enumerate(self.order):
                p = self.people[sid]
                if p["name"]:
                    display = p["name"]
                elif p["is_me"]:
                    display = "You"
                else:
                    display = p["label"]
                people.append({
                    "id": sid, "index": index, "label": p["label"], "name": p["name"], "display": display,
                    "status": p["status"], "evidence": p["evidence"], "alts": list(p["alts"]),
                    "is_me": p["is_me"], "clues": list(p["clues"]), "lines": p["lines"],
                    "last_text": p["last_text"], "source": p["source"],
                })
            return {"rev": self.rev, "now": self.now, "people": people, "log": [dict(l) for l in self.log],
                    "interim": [dict(p) for p in self.interim] if self.interim else None,
                    "proposals": [dict(p) for p in self.proposals if not p.get("applied")][-6:],
                    "corrections": [{"id": c["id"], "line": c["line"], "to": c["to"], "engine": c["engine"]}
                                    for c in self.corrections if not c.get("reverted")][-6:],
                    "events": list(self.events), "assist": self.assist, "notes": self.notes,
                    "attribution": self.attribution,
                    "suggestions": [dict(s, voices=list(s["voices"])) for s in self.suggestions]}
