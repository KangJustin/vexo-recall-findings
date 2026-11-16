"""Gemini's second opinion: merge rules (FakeGemini, canned JSON) and the HTTP client (a local fake server).
No network and no real key are used."""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from familiar import gemini  # noqa: E402
from familiar.resolver import Resolver  # noqa: E402

FAKE_KEY = "test-key-not-real-123"


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        self.t += 3
        return self.t


def convo(*lines):
    r = Resolver(clock=Clock())
    for speaker, text in lines:
        r.feed(speaker, text)
    return r


def person(r, sid):
    return next(p for p in r.snapshot()["people"] if p["id"] == sid)


def says(voice, name, confidence, evidence, wearer=False):
    return {"voice": voice, "name": name, "confidence": confidence, "evidence": evidence, "is_wearer": wearer}


def reply(people=(), same=()):
    return {"people": list(people), "same_person": list(same)}


class FakeGemini:
    """Returns canned replies in order (the last one repeats), or raises, or waits first."""

    def __init__(self, *replies, error=None, delay=0.0):
        self.replies, self.error, self.delay = list(replies), error, delay
        self.seen = []

    def ask(self, data):
        self.seen.append(data)
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class Names(unittest.TestCase):
    def test_gemini_fixes_a_missed_name(self):
        r = convo((0, "Who's this?"), (1, "Oh, people know me as Rafa around here."))
        self.assertIsNone(person(r, 1)["name"])                      # the regexes miss it
        r.apply_model_opinion(reply([says(1, "Rafa", 0.92, "people know me as Rafa")]))
        p = person(r, 1)
        self.assertEqual((p["name"], p["status"], p["source"]), ("Rafa", "heard", "gemini"))
        self.assertIn("people know me as Rafa", p["evidence"])
        self.assertEqual(r.snapshot()["events"][-1]["source"], "gemini")

    def test_middle_confidence_is_possible(self):
        r = convo((0, "Thanks Rafa."), (1, "No problem."))
        r.apply_model_opinion(reply([says(1, "Rafa", 0.65, "Thanks Rafa")]))
        self.assertEqual(person(r, 1)["status"], "possible")

    def test_low_confidence_ignored(self):
        r = convo((0, "Who's this?"), (1, "Oh, people know me as Rafa."))
        r.apply_model_opinion(reply([says(1, "Rafa", 0.45, "people know me as Rafa")]))
        self.assertIsNone(person(r, 1)["name"])

    def test_typed_name_wins(self):
        r = convo((1, "Hi, I'm Maya."), (1, "Well, my friends call me Mara."))
        r.set_name(1, "Maya Chen")
        r.apply_model_opinion(reply([says(1, "Mara", 0.97, "call me Mara")]))
        p = person(r, 1)
        self.assertEqual((p["name"], p["status"], p["source"]), ("Maya Chen", "confirmed", "typed"))
        self.assertIn("Mara", p["alts"])

    def test_evidence_must_be_in_transcript(self):
        r = convo((0, "Who's this?"), (1, "Oh, people know me as Rafa."))
        r.apply_model_opinion(reply([says(1, "Rafa", 0.95, "my name is Rafa")]))      # never said
        self.assertIsNone(person(r, 1)["name"])

    def test_evidence_must_contain_the_name(self):
        r = convo((0, "Who's this?"), (1, "Oh, people know me as Rafa."))
        r.apply_model_opinion(reply([says(1, "Ravi", 0.95, "people know me as")]))
        self.assertIsNone(person(r, 1)["name"])

    def test_evidence_match_ignores_case_quotes_and_spaces(self):
        r = convo((0, "Who's this?"), (1, "Oh, people know me as Rafa."))
        r.apply_model_opinion(reply([says(1, "Rafa", 0.9, "“People  know me as rafa”")]))
        self.assertEqual(person(r, 1)["name"], "Rafa")

    def test_rejected_name_never_reapplied(self):
        r = convo((1, "Hi, I'm Maya."))
        r.clear_name(1)
        r.apply_model_opinion(reply([says(1, "Maya", 0.99, "I'm Maya")]))
        r.apply_model_opinion(reply([says(1, "maya", 0.99, "I'm Maya")]))
        self.assertIsNone(person(r, 1)["name"])

    def test_never_replaces_a_heard_name(self):
        r = convo((1, "Hi, I'm Maya."), (0, "Nice to meet you, Mia."))
        r.apply_model_opinion(reply([says(1, "Mia", 0.9, "Nice to meet you, Mia")]))
        p = person(r, 1)
        self.assertEqual((p["name"], p["source"]), ("Maya", "regex"))
        self.assertIn("Mia", p["alts"])

    def test_upgrades_a_possible_regex_name(self):
        r = convo((1, "Where's the coffee?"), (0, "Over there. Hi Dev, good to see you."))
        self.assertEqual((person(r, 1)["status"], person(r, 1)["source"]), ("possible", "regex"))
        r.apply_model_opinion(reply([says(1, "Dev", 0.88, "Hi Dev")]))
        self.assertEqual((person(r, 1)["status"], person(r, 1)["source"]), ("heard", "gemini"))

    def test_mentioned_name_is_not_an_identity(self):
        r = convo((0, "How was the trip?"), (1, "Great. I talked to Maya about the robot yesterday."))
        r.apply_model_opinion(reply([says(1, "Maya", 0.95, "I talked to Maya")]))
        self.assertIsNone(person(r, 1)["name"])

    def test_name_said_by_an_unresolved_line_does_not_count(self):
        r = Resolver(clock=Clock())
        r.feed(0, "Who's this?", meta={"start": 0, "end": 1, "words": [], "status": "provider"})
        r.feed(None, "Oh, people know me as Rafa.", meta={"start": 1, "end": 3, "words": [], "status": "uncertain"})
        r.feed(1, "Anyway.", meta={"start": 3, "end": 4, "words": [], "status": "provider"})
        r.apply_model_opinion(reply([says(1, "Rafa", 0.95, "people know me as Rafa")]))
        self.assertIsNone(person(r, 1)["name"])

    def test_null_name_never_clears(self):
        r = convo((1, "Hi, I'm Maya."))
        r.apply_model_opinion(reply([says(1, None, 0.9, "")]))
        self.assertEqual(person(r, 1)["name"], "Maya")

    def test_unknown_voice_and_garbage_are_ignored(self):
        r = convo((1, "Hi, I'm Maya."))
        for bad in [None, [], "text", {"people": "x"}, {"people": [None, 3, {"voice": "x"}, says(9, "Zed", 1, "Maya")]},
                    {"people": [{"voice": 1, "name": 5, "confidence": "high"}]}, {"same_person": [{"voices": [1]}]}]:
            r.apply_model_opinion(bad)
        self.assertEqual([p["id"] for p in r.snapshot()["people"]], [1])
        self.assertEqual(person(r, 1)["name"], "Maya")

    def test_reply_for_an_old_conversation_is_dropped(self):
        r = convo((0, "Who's this?"), (1, "Oh, people know me as Rafa."))
        old = r.model_input()["epoch"]
        r.reset()
        r.feed(0, "Who's this?")
        r.feed(1, "Oh, people know me as Rafa.")
        r.apply_model_opinion(reply([says(1, "Rafa", 0.9, "people know me as Rafa")]), epoch=old)
        self.assertIsNone(person(r, 1)["name"])

    def test_wearer_only_when_nobody_said(self):
        r = convo((0, "What's your name?"), (1, "Priya."))
        r.apply_model_opinion(reply([says(0, None, 0.1, "", wearer=True)]))
        self.assertTrue(person(r, 0)["is_me"])
        r2 = convo((0, "What's your name?"), (1, "Priya."))
        r2.set_me(1)
        r2.apply_model_opinion(reply([says(0, None, 0.1, "", wearer=True)]))
        self.assertFalse(person(r2, 0)["is_me"])
        self.assertTrue(person(r2, 1)["is_me"])

    def test_model_input(self):
        r = convo((0, "What's your name?"), (1, "Priya."))
        r.set_name(0, "Alex")
        r.feed(2, "Hi, I'm Sam.")
        r.clear_name(2)
        data = r.model_input()
        self.assertEqual(data["lines"][-1], {"voice": 2, "text": "Hi, I'm Sam."})
        self.assertEqual(data["rejected"], {2: ["Sam"]})
        self.assertTrue(next(p for p in data["people"] if p["voice"] == 0)["typed"])
        prompt = gemini.build_prompt(data)
        self.assertIn("Voice 1: Priya.", prompt)
        self.assertIn("- Voice 2: Sam", prompt)
        self.assertIn('name="Alex", status=confirmed, typed_by_user=true', prompt)
        self.assertIsNone(Resolver().model_input())


class SamePerson(unittest.TestCase):
    def test_sure_and_never_overlapped_auto_merges(self):
        r = convo((1, "Hi, I'm Maya."), (0, "Cool."), (2, "and I work at Northline."))
        r.apply_model_opinion(reply(same=[{"voices": [2, 1], "confidence": 0.9, "reason": "sentence continues"}]))
        ids = [p["id"] for p in r.snapshot()["people"]]
        self.assertNotIn(2, ids)
        self.assertIn("Works at Northline", person(r, 1)["clues"])
        self.assertEqual(r.snapshot()["events"][-1]["kind"], "merge")
        self.assertEqual(r.snapshot()["suggestions"], [])

    def test_overlapped_voices_only_ask(self):
        r = Resolver()
        r.feed(1, "So I was", final=False, t=10)
        r.feed(2, "Yeah", final=True, t=11)                       # while voice 1 was still talking
        r.feed(1, "So I was saying.", final=True, t=12)
        self.assertTrue(r.overlapped(1, 2))
        r.apply_model_opinion(reply(same=[{"voices": [1, 2], "confidence": 0.95, "reason": "x"}]))
        self.assertEqual(len(r.snapshot()["people"]), 2)
        self.assertEqual(r.snapshot()["suggestions"][0]["voices"], [1, 2])

    def test_less_sure_asks_and_yes_merges(self):
        r = convo((1, "Hi, I'm Maya."), (2, "I design things."))
        r.apply_model_opinion(reply(same=[{"voices": [1, 2], "confidence": 0.7, "reason": "same topic"}]))
        sug = r.snapshot()["suggestions"]
        self.assertEqual(len(sug), 1)
        r.apply_model_opinion(reply(same=[{"voices": [2, 1], "confidence": 0.7, "reason": "again"}]))
        self.assertEqual(len(r.snapshot()["suggestions"]), 1)    # no duplicate question
        r.answer_suggestion(sug[0]["id"], True)
        self.assertEqual([p["id"] for p in r.snapshot()["people"]], [1])   # kept the one with the name
        self.assertEqual(r.snapshot()["suggestions"], [])

    def test_no_means_never_ask_again(self):
        r = convo((1, "Hi, I'm Maya."), (2, "I design things."))
        r.apply_model_opinion(reply(same=[{"voices": [1, 2], "confidence": 0.7, "reason": ""}]))
        r.answer_suggestion(r.snapshot()["suggestions"][0]["id"], False)
        r.apply_model_opinion(reply(same=[{"voices": [1, 2], "confidence": 0.99, "reason": ""}]))
        self.assertEqual(len(r.snapshot()["people"]), 2)
        self.assertEqual(r.snapshot()["suggestions"], [])

    def test_two_different_heard_names_are_never_auto_merged(self):
        r = convo((1, "Hi, I'm Maya."), (2, "I'm Dev."))
        r.apply_model_opinion(reply(same=[{"voices": [1, 2], "confidence": 0.99, "reason": ""}]))
        self.assertEqual(len(r.snapshot()["people"]), 2)
        self.assertEqual(len(r.snapshot()["suggestions"]), 1)

    def test_weak_suggestion_ignored(self):
        r = convo((1, "Hi."), (2, "Hello."))
        r.apply_model_opinion(reply(same=[{"voices": [1, 2], "confidence": 0.3, "reason": ""}]))
        self.assertEqual(r.snapshot()["suggestions"], [])

    def test_manual_merge_updates_pending_question(self):
        r = convo((1, "Hi."), (2, "Hello."), (3, "Hey."))
        r.apply_model_opinion(reply(same=[{"voices": [1, 2], "confidence": 0.6, "reason": ""}]))
        r.merge(2, 1)                                            # the user did it by hand
        self.assertEqual(r.snapshot()["suggestions"], [])


class AdvisorTests(unittest.TestCase):
    def test_success(self):
        r = convo((0, "Who's this?"), (1, "Oh, people know me as Rafa."))
        fake = FakeGemini(reply([says(1, "Rafa", 0.9, "people know me as Rafa")]))
        advisor = gemini.Advisor(r, fake, log=lambda _: None)
        self.assertTrue(advisor.run_once())
        self.assertEqual(person(r, 1)["name"], "Rafa")
        self.assertEqual(r.snapshot()["assist"], "on")

    def test_errors_fall_back_to_rules(self):
        for error in [gemini.GeminiError("Gemini took longer than 4 s."), TimeoutError(), ValueError("boom"), KeyError("x")]:
            r = convo((1, "Hi, I'm Maya."))
            said = []
            advisor = gemini.Advisor(r, FakeGemini(error=error), log=said.append)
            self.assertFalse(advisor.run_once())
            self.assertEqual(person(r, 1)["name"], "Maya")         # the regex result stays
            self.assertEqual(r.snapshot()["assist"], "rules")
            self.assertEqual(len(said), 1)
            advisor.run_once()
            self.assertEqual(len(said), 1)                         # the same problem is said once
            advisor.client = FakeGemini(reply())
            self.assertTrue(advisor.run_once())
            self.assertEqual(r.snapshot()["assist"], "on")

    def test_backoff_after_failures(self):
        now = [0.0]
        advisor = gemini.Advisor(convo((1, "Hi.")), FakeGemini(error=TimeoutError()), interval=2.0,
                                 clock=lambda: now[0], log=lambda _: None)
        advisor.run_once()
        advisor.run_once()
        self.assertEqual(advisor._next, 8.0)                       # 2 s * 2^2 after two failures

    def test_slow_gemini_never_blocks_and_is_debounced(self):
        r = Resolver()
        fake = FakeGemini(reply(), delay=0.3)
        advisor = gemini.Advisor(r, fake, interval=0.4, log=lambda _: None).start()
        try:
            started = time.monotonic()
            for i in range(8):
                r.feed(i % 2, "line %d" % i)
            self.assertLess(time.monotonic() - started, 0.1)       # feeding never waits for Gemini
            time.sleep(1.0)
        finally:
            advisor.stop()
        self.assertGreaterEqual(len(fake.seen), 1)
        self.assertLessEqual(len(fake.seen), 3)                    # 8 lines, at most one call per 0.4 s
        self.assertEqual(fake.seen[-1]["lines"][-1]["text"], "line 7")


# ---------------------------------------------------------------- the HTTP client, against a local fake Gemini

class FakeServer:
    def __init__(self, respond):
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
                code, payload, delay = respond(body, len(outer.requests))
                time.sleep(delay)
                data = json.dumps(payload).encode()
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d/v1beta" % self.server.server_address[1]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def interaction(obj):
    return {"id": "x", "status": "completed", "steps": [
        {"type": "thought", "summary": []},
        {"type": "model_output", "content": [{"type": "text", "text": json.dumps(obj)}]}]}


DATA = {"epoch": 1, "lines": [{"voice": 1, "text": "Hi, I'm Maya."}], "people": [], "rejected": {}}


class Client(unittest.TestCase):
    def serve(self, respond):
        server = FakeServer(respond)
        self.addCleanup(server.close)
        return server, gemini.GeminiClient(FAKE_KEY, model="test-model", timeout=0.5, base_url=server.url)

    def test_request_shape_and_reply(self):
        want = reply([says(1, "Maya", 0.9, "I'm Maya")])
        server, client = self.serve(lambda body, n: (200, interaction(want), 0))
        self.assertEqual(client.ask(DATA), want)
        req = server.requests[0]
        self.assertEqual(req["path"], "/v1beta/interactions")
        self.assertEqual(req["headers"]["x-goog-api-key"], FAKE_KEY)
        body = req["body"]
        self.assertEqual(body["model"], "test-model")
        self.assertIs(body["store"], False)
        self.assertEqual(body["response_format"]["mime_type"], "application/json")
        self.assertEqual(body["response_format"]["schema"], gemini.SCHEMA)
        self.assertIn("Voice 1: Hi, I'm Maya.", body["input"])
        self.assertIn("never the speaker", body["system_instruction"])
        self.assertNotIn(FAKE_KEY, json.dumps(body))               # the key is only in the header

    def test_http_error_is_short_and_has_no_key(self):
        server, client = self.serve(lambda body, n: (429, {"error": {"message": "quota " + FAKE_KEY}}, 0))
        with self.assertRaises(gemini.GeminiError) as caught:
            client.ask(DATA)
        self.assertEqual(str(caught.exception), "Gemini answered HTTP 429.")
        self.assertNotIn(FAKE_KEY, repr(client))

    def test_400_retries_once_without_optional_fields(self):
        server, client = self.serve(lambda body, n: (400, {}, 0) if "generation_config" in body else
                                    (200, interaction(reply()), 0))
        self.assertEqual(client.ask(DATA), reply())
        self.assertEqual(len(server.requests), 2)
        self.assertIn("never the speaker", server.requests[1]["body"]["input"])
        client.ask(DATA)
        self.assertEqual(len(server.requests), 3)                  # stays lean, no second retry

    def test_timeout(self):
        server, client = self.serve(lambda body, n: (200, interaction(reply()), 1.5))
        started = time.monotonic()
        with self.assertRaises(gemini.GeminiError) as caught:
            client.ask(DATA)
        self.assertLess(time.monotonic() - started, 1.2)
        self.assertIn("longer than", str(caught.exception))

    def test_bad_replies(self):
        for payload in [{"status": "failed"}, interaction("not an object"),
                        {"status": "completed", "steps": [{"type": "model_output", "content": [{"type": "text", "text": "{oops"}]}]}]:
            server, client = self.serve(lambda body, n, p=payload: (200, p, 0))
            with self.assertRaises(gemini.GeminiError):
                client.ask(DATA)

    def test_unreachable(self):
        client = gemini.GeminiClient(FAKE_KEY, timeout=0.5, base_url="http://127.0.0.1:9/v1beta")
        with self.assertRaises(gemini.GeminiError):
            client.ask(DATA)

    def test_find_key(self):
        saved = os.environ.pop(gemini.KEY_NAME, None)
        try:
            with tempfile.TemporaryDirectory() as folder:
                self.assertIsNone(gemini.find_key(folders=[folder]))
                with open(os.path.join(folder, ".env"), "w") as f:
                    f.write("DEEPGRAM_API_KEY=other\nexport GEMINI_API_KEY=\"abc\"\n")
                self.assertEqual(gemini.find_key(folders=[folder]), "abc")
                os.environ[gemini.KEY_NAME] = "from-env"
                self.assertEqual(gemini.find_key(folders=[folder]), "from-env")
        finally:
            os.environ.pop(gemini.KEY_NAME, None)
            if saved is not None:
                os.environ[gemini.KEY_NAME] = saved



# ---------------------------------------------------------------- Gemini Decision 1

def decisions_reply(**answers):
    return {"answers": answers, "usage": {"total_input_tokens": 100, "total_output_tokens": 0, "total_tokens": 100}}


def pick(choice, confidence):
    return {"type": "choice", "choice": choice, "probabilities": {}, "confidence": confidence}


def yes(p):
    return {"type": "noul", "noul": p, "confidence": max(p, 1 - p)}


class Decision(unittest.TestCase):
    def body(self, r, **kw):
        return gemini.DecisionClient(FAKE_KEY, **kw).body(r.model_input())

    def test_candidates(self):
        lines = [{"voice": 0, "text": "Over by the door. Hi Dev, good to see you."},
                 {"voice": 1, "text": "Oh, people know me as Rafa. I work at Northline."},
                 {"voice": 0, "text": "Thanks. Sorry, Monday works."}]
        names = gemini.candidate_names(lines)
        for name in ["Dev", "Rafa", "Northline"]:
            self.assertIn(name, names)
        for word in ["Over", "Oh", "Thanks", "Sorry", "Monday", "I"]:
            self.assertNotIn(word, names)
        self.assertEqual(set(names[:2]), {"Northline", "Rafa"})      # most recently heard first

    def test_request_questions(self):
        r = convo((0, "Who's this?"), (1, "Oh, people know me as Rafa."), (2, "Hi, I'm Sam."), (3, "Hello."))
        r.set_name(0, "Alex")
        r.clear_name(2)                                              # "Not Sam?"
        r.apply_model_opinion(reply(same=[{"voices": [1, 3], "confidence": 0.6, "reason": ""}]))
        r.answer_suggestion(r.snapshot()["suggestions"][0]["id"], False)
        body = self.body(r)
        q = body["questions"]
        self.assertEqual(body["model"], "gemini-decision-1")
        self.assertNotIn("name_v0", q)                               # typed: never asked
        self.assertIn("Rafa", q["name_v1"]["criteria"])
        self.assertIn(gemini.NO_NAME, q["name_v1"]["criteria"])
        self.assertNotIn("Sam", q["name_v2"]["criteria"])            # rejected for this voice: not even offered
        self.assertIn("Sam", q["name_v1"]["criteria"])
        self.assertNotIn("same_v1_v3", q)                            # the user said two people
        self.assertEqual(q["same_v0_v1"]["type"], "noul")
        self.assertIn("wearer", q)                                   # typing a name does not say who wears the band
        self.assertNotIn("Alex", q["name_v1"]["criteria"])           # typed for voice 0: not offered to others
        for key in q:
            self.assertRegex(key, r"^[A-Za-z0-9_-]{1,64}$")
        self.assertIn("Voice 1: Oh, people know me as Rafa.", body["input"])
        self.assertIn("Voice 0 is Alex", body["input"])
        self.assertIn("Voice 2 is NOT named Sam", body["input"])

    def test_wearer_question_only_when_unknown(self):
        r = convo((0, "What's your name?"), (1, "Priya."))
        self.assertIn("wearer", self.body(r)["questions"])
        r.set_me(0)
        self.assertNotIn("wearer", self.body(r)["questions"])

    def test_nothing_to_ask(self):
        self.assertIsNone(gemini.DecisionClient(FAKE_KEY).body({"lines": [], "people": []}))
        r = convo((1, "Hello."))                                     # one voice, no candidate names
        r.set_me(1)
        self.assertIsNone(self.body(r))

    def test_answers_become_names_through_the_merge_rules(self):
        r = convo((0, "Who's this?"), (1, "Oh, people know me as Rafa around here."), (2, "Hey."), (0, "Hi Dev."))
        body = self.body(r)
        answers = {"name_v1": pick("Rafa", 0.93), "name_v2": pick("Dev", 0.7), "name_v0": pick(gemini.NO_NAME, 0.9),
                   "wearer": pick("voice_0", 0.85), "same_v0_v1": yes(0.02), "same_v0_v2": yes(0.03),
                   "same_v1_v2": yes(0.04)}
        r.apply_model_opinion(gemini.DecisionClient.convert(body, answers, r.model_input()["lines"]))
        self.assertEqual((person(r, 1)["name"], person(r, 1)["status"], person(r, 1)["source"]), ("Rafa", "heard", "gemini"))
        self.assertIn("people know me as Rafa", person(r, 1)["evidence"])
        self.assertEqual((person(r, 2)["name"], person(r, 2)["status"]), ("Dev", "possible"))
        self.assertIsNone(person(r, 0)["name"])
        self.assertTrue(person(r, 0)["is_me"])
        self.assertEqual(r.snapshot()["suggestions"], [])

    def test_unsure_and_odd_answers(self):
        r = convo((0, "Who's this?"), (1, "Oh, people know me as Rafa."))
        body = self.body(r)
        for answers in [{"name_v1": pick("Rafa", 0.4)}, {"name_v1": pick("Zed", 0.99)}, {"name_v1": "x"},
                        {"name_v1": {"choice": None}}, {"wearer": pick("voice_x", 0.99)}, {}]:
            r.apply_model_opinion(gemini.DecisionClient.convert(body, answers, r.model_input()["lines"]))
        self.assertIsNone(person(r, 1)["name"])

    def test_same_person_from_noul(self):
        r = convo((1, "Hi, I'm Maya."), (0, "Cool."), (2, "and I work at Northline."))
        body = self.body(r)
        converted = gemini.DecisionClient.convert(body, {"same_v1_v2": yes(0.92)}, r.model_input()["lines"])
        self.assertEqual(converted["same_person"][0]["voices"], [1, 2])
        r.apply_model_opinion(converted)
        self.assertNotIn(2, [p["id"] for p in r.snapshot()["people"]])

    def test_over_http(self):
        def respond(body, n):
            return 200, decisions_reply(name_v1=pick("Rafa", 0.9), same_v0_v1=yes(0.01), wearer=pick("voice_0", 0.6)), 0
        server = FakeServer(respond)
        self.addCleanup(server.close)
        r = convo((0, "Who's this?"), (1, "Oh, people know me as Rafa."))
        client = gemini.DecisionClient(FAKE_KEY, timeout=0.5, base_url=server.url)
        self.assertTrue(gemini.Advisor(r, client, log=lambda _: None).run_once())
        req = server.requests[0]
        self.assertEqual(req["path"], "/v1beta/decisions")
        self.assertEqual(req["headers"]["x-goog-api-key"], FAKE_KEY)
        self.assertNotIn(FAKE_KEY, json.dumps(req["body"]))
        self.assertEqual(person(r, 1)["name"], "Rafa")
        self.assertFalse(person(r, 0)["is_me"])                      # wearer confidence 0.6 < 0.8: not used

    def test_http_problems(self):
        r = convo((0, "Who's this?"), (1, "Oh, people know me as Rafa."))
        for code, payload, delay in [(400, {"error": {"code": "safety"}}, 0), (200, {"usage": {}}, 0),
                                     (200, decisions_reply(), 1.5)]:
            server = FakeServer(lambda body, n, c=code, p=payload, d=delay: (c, p, d))
            self.addCleanup(server.close)
            with self.assertRaises(gemini.GeminiError):
                gemini.DecisionClient(FAKE_KEY, timeout=0.5, base_url=server.url).ask(r.model_input())

    def test_engine_choice(self):
        self.assertIsInstance(gemini.make_client(FAKE_KEY), gemini.DecisionClient)
        self.assertIsInstance(gemini.make_client(FAKE_KEY, "text"), gemini.GeminiClient)
        self.assertNotIn(FAKE_KEY, repr(gemini.make_client(FAKE_KEY)))


class Notes(unittest.TestCase):
    LINES = [{"id": 1, "speaker": 0, "text": "What are you building?", "status": "provider"},
             {"id": 2, "speaker": 1, "text": "An arm. Maya helped with the camera.", "status": "provider"},
             {"id": 3, "speaker": None, "text": "I'll send the files tonight.", "status": "uncertain"},
             {"id": 4, "speaker": 1, "text": "Let's test it Tuesday.", "status": "uncertain"}]

    def test_attribution_comes_from_our_lines_not_the_model(self):
        raw = {"topics": ["robot arm"], "mentioned_names": [{"name": "Maya", "line_id": 2}, {"name": "Zed", "line_id": 2},
                                                             {"name": "Maya", "line_id": 99}],
               "follow_ups": [{"text": "send the files", "line_id": 3}, {"text": "test Tuesday", "line_id": 4},
                              {"text": "ask about the arm", "line_id": 1}]}
        out = gemini.validate_notes(raw, self.LINES)
        self.assertEqual(out["mentioned_names"], [{"name": "Maya", "line_id": 2, "said_by": 1,
                                                   "quote": "An arm. Maya helped with the camera."}])
        self.assertEqual([(f["line_id"], f["owner"]) for f in out["follow_ups"]], [(3, None), (4, None), (1, 0)])

    def test_notes_call_and_worker(self):
        from unittest import mock
        client = mock.Mock()
        client.interactions.create.return_value = mock.Mock(output_text=json.dumps(
            {"topics": ["arm"], "mentioned_names": [], "follow_ups": []}))
        book = Resolver()
        for who, text in [(0, "Hi there."), (1, "Hello."), (0, "Nice arm.")]:
            book.feed(who, text)
        worker = gemini.NotesWorker(book, gemini.ConversationNotes(client=client, model="m"), min_new=3, log=lambda _: None)
        self.assertTrue(worker.run_once())
        self.assertEqual(book.snapshot()["notes"]["topics"], ["arm"])
        self.assertFalse(worker.run_once())                       # nothing new since
        sent = client.interactions.create.call_args.kwargs
        self.assertEqual(sent["response_format"]["schema"], gemini.NOTES_SCHEMA)
        self.assertIn("L1 [person 0]: Hi there.", sent["input"])

if __name__ == "__main__":
    unittest.main()
