"""Turns a diarized conversation into "who is who".

The speech service tells us WHEN someone different is talking ("voice 0", "voice 1", ...)
but not WHO they are. This module listens to what is said and picks up name clues:

  * self introductions        "Hi, I'm Maya"  /  "my name is Maya Chen"  /  "call me Dev"
  * an answer to a question   "What's your name?"  ->  "Priya."
  * being addressed           "Nice to meet you, Dev"  ->  names the person who spoke just before
  * small facts about them    "I work at Northline", "I'm a product designer"

Anything it is not sure about is shown as "possible" so the person using the app can
confirm, fix or fill in the name by hand. A name typed by the user always wins.

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
        if notify:
            self._changed()

    def _person(self, sid):
        sid = self.alias.get(sid, sid)
        if sid not in self.people:
            self.order.append(sid)
            self.people[sid] = {
                "id": sid, "label": "Voice " + chr(65 + (len(self.order) - 1) % 26),
                "name": None, "status": "unknown", "evidence": None, "alts": [],
                "rejected": [], "is_me": False, "clues": [], "lines": 0, "last_text": "",
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
                 "name": person["name"], "status": person["status"], "why": why}
        self.events = (self.events + [event])[-12:]
        if self.on_event:
            self.on_event(event)

    # ------------------------------------------------------------------ input

    def feed(self, speaker, text, final=True, t=None):
        """One piece of speech. speaker: the voice number from diarization (None = same as before)."""
        text = _norm(text)
        if not text:
            return
        now = self._clock() if t is None else t
        with self._lock:
            if speaker is None:
                speaker = self.now if self.now is not None else 0
            speaker = self.alias.get(speaker, speaker)
            person = self._person(speaker)
            self.now = person["id"]
            if not final:
                self.interim = {"speaker": person["id"], "text": text}
            else:
                self.interim = None
                self._final(person, text, now)
        self._changed()

    def _final(self, person, text, now):
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
            self._offer(person, self_name, "heard", why)

        addressed = find_addressed_name(text)
        if addressed:
            target = self._recent_other(sid, now)
            if target is not None and (not self_name or addressed != self_name):
                target_person = self._person(target)
                if addressed != person["name"]:
                    self._offer(target_person, addressed, "possible", "was called “%s” by %s" % (
                        addressed, "you" if person["is_me"] else (person["name"] or person["label"])))

        # 2. small facts the speaker says about themselves
        for clue in extract_clues(text):
            if clue.lower() not in [c.lower() for c in person["clues"]]:
                person["clues"] = (person["clues"] + [clue])[-5:]

        person["lines"] += 1
        person["last_text"] = text
        self.history = (self.history + [(sid, now)])[-30:]
        self._line_id += 1
        self.log = (self.log + [{"id": self._line_id, "speaker": sid, "text": text}])[-60:]

    def _recent_other(self, sid, now):
        for speaker, when in reversed(self.history):
            if now - when > self._window:
                break
            if speaker != sid:
                return speaker
        return None

    def _offer(self, person, name, status, why):
        """A name candidate. Typed names win; heard beats possible; first heard name sticks."""
        if name in person["rejected"]:
            return
        if person["status"] == "confirmed":
            if name != person["name"] and name not in person["alts"]:
                person["alts"].append(name)
            return
        if person["name"] and person["name"] == name:
            if STATUS_RANK[status] > STATUS_RANK[person["status"]]:
                person["status"], person["evidence"] = status, why
            return
        if STATUS_RANK[status] > STATUS_RANK[person["status"]] or not person["name"]:
            if person["name"] and person["name"] not in person["alts"]:
                person["alts"].append(person["name"])
            person["name"], person["status"], person["evidence"] = name, status, why
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
            if name in person["rejected"]:
                person["rejected"].remove(name)
        self._changed()

    def clear_name(self, speaker):
        """"Not Maya?": forget the name, and do not offer it for this voice again."""
        with self._lock:
            person = self._person(speaker)
            if person["name"] and person["name"] not in person["rejected"]:
                person["rejected"].append(person["name"])
            person["name"], person["status"], person["evidence"] = None, "unknown", None
        self._changed()

    def set_me(self, speaker, flag=True):
        with self._lock:
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
            if not b["name"] and a["name"]:
                b["name"], b["status"], b["evidence"] = a["name"], a["status"], a["evidence"]
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
                    "last_text": p["last_text"],
                })
            return {"rev": self.rev, "now": self.now, "people": people, "log": list(self.log),
                    "interim": dict(self.interim) if self.interim else None,
                    "events": list(self.events)}
