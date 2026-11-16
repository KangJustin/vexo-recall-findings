"""Second pass, Gemini review validation, overlap evidence and reversible corrections. No network: engines are fakes."""
import json
import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from familiar import second_pass as sp  # noqa: E402
from familiar.resolver import Resolver  # noqa: E402


def line(book, speaker, words, status="provider"):
    """Add a settled line: words [(text, start, end), ...]."""
    book.feed(speaker, " ".join(w[0] for w in words), meta={"start": words[0][1], "end": words[-1][2],
                                                             "words": words, "status": status, "label": "1:%s" % speaker})
    return book.snapshot()["log"][-1]


def ew(text, start, end, speaker, conf=None):
    return {"text": text, "start": start, "end": end, "speaker": speaker, "speaker_confidence": conf}


class Overlap(unittest.TestCase):
    def test_overlapping_words_of_two_speakers_are_marked(self):
        book = Resolver()
        line(book, 0, [("I", 1.0, 1.2), ("think", 1.2, 1.6), ("the", 1.6, 1.8), ("camera", 1.8, 2.4)])
        line(book, 1, [("no", 1.7, 2.0), ("no", 2.0, 2.3)])          # 0.6 s of words at the same time
        log = book.snapshot()["log"]
        self.assertEqual(log[1]["overlaps"][0]["with"], 0)
        self.assertEqual(log[0]["overlaps"][0]["with"], 1)

    def test_timestamp_jitter_is_not_overlap(self):
        book = Resolver()
        line(book, 0, [("okay", 1.0, 1.42)])
        line(book, 1, [("sure", 1.38, 1.7)])                         # 0.04 s: jitter
        self.assertEqual(book.snapshot()["log"][1]["overlaps"], [])

    def test_same_speaker_is_never_overlap(self):
        book = Resolver()
        line(book, 0, [("one", 1.0, 1.5)])
        line(book, 0, [("two", 1.2, 1.8)])
        self.assertEqual(book.snapshot()["log"][1]["overlaps"], [])


class Unresolved(unittest.TestCase):
    def test_unresolved_line_has_no_person(self):
        book = Resolver()
        line(book, 0, [("hello", 0.0, 0.5)])
        book.feed(None, "Hi, I'm Maya.", meta={"start": 1, "end": 2, "words": [["Hi,", 1, 1.2]], "status": "uncertain",
                                               "candidates": [0]})
        snap = book.snapshot()
        self.assertEqual([p["id"] for p in snap["people"]], [0])
        self.assertIsNone(snap["log"][-1]["speaker"])
        self.assertEqual(snap["log"][-1]["candidates"], [0])
        self.assertIsNone(snap["people"][0]["name"])                 # an unresolved intro names nobody


class Corrections(unittest.TestCase):
    def setUp(self):
        self.book = Resolver()
        line(self.book, 0, [("hello", 0.0, 0.4)])
        self.l = line(self.book, 1, [("the", 1.0, 1.2), ("seam", 1.2, 1.5), ("wait", 1.55, 1.8), ("is", 1.9, 2.0),
                                     ("sanded", 2.0, 2.4)])

    def test_apply_splits_the_line_and_revert_restores_it(self):
        self.book.add_proposals([{"engine": "test", "line": self.l["id"], "words": [2, 3], "start": 1.55, "end": 1.8,
                                  "from": 1, "to": 0, "confidence": 0.95, "reason": "x"}])
        pid = self.book.proposals[-1]["id"]
        self.assertFalse(self.book.proposals[-1]["applied"])          # suggestions are not applied by themselves
        done = self.book.apply_proposal(pid)
        texts = [(l["speaker"], l["text"], l["status"]) for l in self.book.snapshot()["log"][1:]]
        self.assertEqual(texts, [(1, "the seam", "provider"), (0, "wait", "corrected"), (1, "is sanded", "provider")])
        self.assertTrue(self.book.revert_correction(done["id"]))
        self.assertEqual([l["text"] for l in self.book.snapshot()["log"]], ["hello", "the seam wait is sanded"])
        self.assertFalse(self.book.proposals[-1]["applied"])

    def test_two_proposals_on_one_line_apply_together(self):
        self.book.add_proposals([
            {"engine": "t", "line": self.l["id"], "words": [0, 1], "start": 1.0, "end": 1.2, "from": 1, "to": 0,
             "confidence": 0.95, "reason": ""},
            {"engine": "t", "line": self.l["id"], "words": [4, 5], "start": 2.0, "end": 2.4, "from": 1, "to": None,
             "confidence": 0.95, "reason": ""}])
        self.book.apply_proposals([p["id"] for p in self.book.proposals])
        got = [(l["speaker"], l["text"]) for l in self.book.snapshot()["log"][1:]]
        self.assertEqual(got, [(0, "the"), (1, "seam wait is"), (None, "sanded")])


class Reconcile(unittest.TestCase):
    def session(self):
        book = Resolver()
        line(book, 0, [("so", 10.0, 10.2), ("what", 10.2, 10.4), ("are", 10.4, 10.6), ("you", 10.6, 10.8),
                       ("building", 10.8, 11.2)])
        line(book, 1, [("a", 11.5, 11.6), ("small", 11.6, 11.9), ("arm", 11.9, 12.2), ("that", 12.2, 12.4),
                       ("sorts", 12.4, 12.8)])
        line(book, 0, [("nice", 13.0, 13.3), ("one", 13.3, 13.6), ("really", 13.6, 14.0)], status="voice")
        return book

    def test_labels_map_through_anchors_and_disagreements_become_proposals(self):
        book = self.session()
        session = sp.session_words(book, 9.0, 16.0)
        engine = ([ew(w["text"], w["start"] + 0.01, w["end"] - 0.01, "A", 0.97) for w in session[:5]] +
                  [ew(w["text"], w["start"], w["end"], "B", 0.96) for w in session[5:10]] +
                  [ew(w["text"], w["start"], w["end"], "B", 0.93) for w in session[10:]])  # engine: the last line is B
        pairs = sp.match_words(session, engine)
        mapping = sp.map_labels(session, engine, pairs, risky=[(13.0, 14.0)])
        self.assertEqual(mapping, {"A": 0, "B": 1})
        props = sp.proposals_for(session, engine, pairs, mapping, (9.0, 16.0), edge=0.5)
        self.assertEqual([(p["words"], p["from"], p["to"]) for p in props], [([0, 3], 0, 1)])
        self.assertAlmostEqual(props[0]["confidence"], 0.93, places=2)

    def test_ambiguous_label_stays_unmapped(self):
        book = self.session()
        session = sp.session_words(book, 9.0, 16.0)
        engine = [ew(w["text"], w["start"], w["end"], "X", 0.9) for w in session]          # one label for both people
        mapping = sp.map_labels(session, engine, sp.match_words(session, engine))
        self.assertEqual(mapping, {})

    def test_words_near_the_clip_edges_are_not_corrected(self):
        book = self.session()
        session = sp.session_words(book, 9.0, 16.0)
        engine = [ew(w["text"], w["start"], w["end"], "A" if w["speaker"] == 0 else "B", 0.95) for w in session]
        engine[0]["speaker"] = "B"                                                            # at 10.0-10.2
        pairs = sp.match_words(session, engine)
        props = sp.proposals_for(session, engine, pairs, {"A": 0, "B": 1}, (9.5, 16.0), edge=1.0)
        self.assertEqual(props, [])


class FakeVoices:
    def __init__(self, seconds):
        self._written = int(seconds * sp.SR)

    def cut(self, a, b):
        return np.zeros(int((b - a) * sp.SR), dtype=np.float32)

    def _new_id(self):
        return 7


class Pass(unittest.TestCase):
    def setUp(self):
        self.book = Resolver()
        line(self.book, 0, [("so", 1.0, 1.2), ("what", 1.2, 1.4), ("do", 1.4, 1.6), ("you", 1.6, 1.8), ("do", 1.8, 2.0)])
        self.l = line(self.book, 0, [("I", 2.05, 2.2), ("build", 2.2, 2.5), ("arms", 2.5, 2.9)])   # no pause: risky
        line(self.book, 1, [("cool", 3.5, 3.8), ("very", 3.8, 4.0), ("cool", 4.0, 4.3)])

    def engine_reply(self, conf):
        words = [("so", 1.0, 1.2, 0), ("what", 1.2, 1.4, 0), ("do", 1.4, 1.6, 0), ("you", 1.6, 1.8, 0), ("do", 1.8, 2.0, 0),
                 ("I", 2.05, 2.2, 1), ("build", 2.2, 2.5, 1), ("arms", 2.5, 2.9, 1),
                 ("cool", 3.5, 3.8, 1), ("very", 3.8, 4.0, 1), ("cool", 4.0, 4.3, 1)]
        return {"results": {"channels": [{"alternatives": [{"words": [
            {"word": w, "punctuated_word": w, "start": a, "end": b, "speaker": s, "speaker_confidence": conf}
            for w, a, b, s in words]}]}]}}

    def run_pass(self, conf, apply):
        second = sp.SecondPass(self.book, FakeVoices(12.0), engine="deepgram", apply=apply, threaded=False,
                               log=lambda _: None)
        with mock.patch.object(sp, "deepgram_batch", lambda wav: self.engine_reply(conf)):
            for settled in self.book.snapshot()["log"]:             # the app notes every settled line, in order
                second.note(settled)
            second.flush(final=True)
        return second

    def test_suggestions_only_by_default(self):
        second = self.run_pass(0.97, apply=False)
        props = self.book.snapshot()["proposals"]
        self.assertEqual([(p["line"], p["words"], p["to"]) for p in props], [(self.l["id"], [0, 3], 1)])
        self.assertEqual(self.book.snapshot()["log"][1]["speaker"], 0)          # nothing changed
        self.assertEqual(second.report()["requests"], 1)

    def test_apply_only_when_the_engine_is_sure(self):
        self.run_pass(0.6, apply=True)
        self.assertEqual(self.book.snapshot()["log"][1]["speaker"], 0)
        self.run_pass(0.97, apply=True)
        moved = [l for l in self.book.snapshot()["log"] if l["text"] == "I build arms"][0]
        self.assertEqual((moved["speaker"], moved["status"]), (1, "corrected"))


class GeminiTranscribeParsing(unittest.TestCase):
    def test_documented_shape(self):
        raw = {"steps": [{"type": "model_output", "content": [{"type": "text", "text": "Hi. Wait.", "annotations": [
            {"type": "word_info", "text": "Hi.", "speaker": "spk_1", "start_offset": "0.100s", "end_offset": "0.400s"},
            {"type": "word_info", "text": "Wait.", "speaker": "spk_2", "start_offset": "0.450s", "end_offset": "0.700s"},
            {"type": "other"}]}]}]}
        words = sp.words_from_gemini_transcribe(raw, offset=30.0)
        self.assertEqual([(w["text"], w["start"], w["speaker"]) for w in words],
                         [("Hi.", 30.1, "spk_1"), ("Wait.", 30.45, "spk_2")])
        self.assertIsNone(words[0]["speaker_confidence"])                        # none is given: never auto-applied


class ReviewValidation(unittest.TestCase):
    def setUp(self):
        book = Resolver()
        line(book, 0, [("we", 10.0, 10.2), ("tried", 10.2, 10.5), ("three", 10.5, 10.8)])
        line(book, 0, [("yeah", 10.6, 10.8)])
        self.session = sp.session_words(book, 9.0, 14.0)
        self.window = (9.0, 14.0)

    def raw(self, **change):
        base = {"clip_start_in_session_seconds": 9.0,
                "segments": [{"start_seconds": 10.6, "end_seconds": 10.8, "candidate_session_speaker_ids": [1, 5],
                              "overlap_status": "overlap", "attribution_status": "differs_from_current"}],
                "proposed_corrections": [{"word_ids": ["w3"], "start_seconds": 10.6, "end_seconds": 10.8,
                                          "to_session_speaker_id": 1, "reason": "second voice, quieter"}],
                "evidence_notes": [{"start_seconds": 10.6, "end_seconds": 10.8, "note": "two voices audible"}],
                "unresolved_regions": [{"start_seconds": 11.0, "end_seconds": 11.5, "reason": "noise"}]}
        base.update(change)
        return base

    def test_valid_review_becomes_hypotheses(self):
        out = sp.validate_review(self.raw(), self.window, self.session, [0, 1])
        self.assertEqual([(p["words"], p["to"], p["engine"], p["confidence"]) for p in out["proposals"]],
                         [([0, 1], 1, "gemini_review", None)])
        self.assertEqual(out["segments"][0]["candidate_session_speaker_ids"], [1])   # unknown id 5 removed
        self.assertEqual(len(out["unresolved"]), 1)

    def test_wrong_offset_rejects_everything(self):
        out = sp.validate_review(self.raw(clip_start_in_session_seconds=0.0), self.window, self.session, [0, 1])
        self.assertEqual(out["proposals"], [])
        self.assertIn("wrong clip offset", out["dropped"])

    def test_bad_corrections_are_dropped(self):
        bad = [{"word_ids": ["w3"], "start_seconds": 20.0, "end_seconds": 21.0, "to_session_speaker_id": 1, "reason": ""},
               {"word_ids": ["w3"], "start_seconds": 10.6, "end_seconds": 10.8, "to_session_speaker_id": 9, "reason": ""},
               {"word_ids": ["w0"], "start_seconds": 12.0, "end_seconds": 13.0, "to_session_speaker_id": 1, "reason": ""},
               {"word_ids": ["w99"], "start_seconds": 10.6, "end_seconds": 10.8, "to_session_speaker_id": 1, "reason": ""}]
        out = sp.validate_review(self.raw(proposed_corrections=bad), self.window, self.session, [0, 1])
        self.assertEqual(out["proposals"], [])
        self.assertEqual(len(out["dropped"]), 4)

    def test_null_speaker_means_unresolved(self):
        fix = [{"word_ids": ["w3"], "start_seconds": 10.6, "end_seconds": 10.8, "to_session_speaker_id": None,
                "reason": "not sure who"}]
        out = sp.validate_review(self.raw(proposed_corrections=fix), self.window, self.session, [0, 1])
        self.assertIsNone(out["proposals"][0]["to"])


class FakeFiles:
    def __init__(self):
        self.uploads = 0

    def upload(self, file, config=None):
        self.uploads += 1
        return mock.Mock(uri="files/abc", mime_type="audio/wav", name="files/abc")


class Review(unittest.TestCase):
    def test_review_uses_schema_and_reuses_uploads(self):
        client = mock.Mock()
        client.files = FakeFiles()
        reply = {"clip_start_in_session_seconds": 0.0, "segments": [], "proposed_corrections": [],
                 "evidence_notes": [], "unresolved_regions": []}
        client.interactions.create.return_value = mock.Mock(output_text=json.dumps(reply), usage=None)
        review = sp.GeminiReview(client=client, model="m", use_files=True)
        audio = np.zeros(sp.SR, dtype=np.float32)
        for _ in range(2):
            review.review(audio, (0.0, 1.0), [], [0, 1])
        self.assertEqual((client.files.uploads, review.stats["reused_uploads"]), (1, 1))
        kwargs = client.interactions.create.call_args.kwargs
        self.assertEqual(kwargs["response_format"]["schema"], sp.REVIEW_SCHEMA)
        self.assertEqual(kwargs["input"][0]["uri"], "files/abc")
        self.assertIn("never tells you who is speaking", kwargs["system_instruction"])

    def test_prose_reply_is_an_error(self):
        client = mock.Mock()
        client.interactions.create.return_value = mock.Mock(output_text="Speaker 1 says hi", usage=None)
        with self.assertRaises(sp.SecondPassError):
            sp.GeminiReview(client=client, model="m").review(np.zeros(1600, np.float32), (0.0, 0.1), [], [0])


if __name__ == "__main__":
    unittest.main()
