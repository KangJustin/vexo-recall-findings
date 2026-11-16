"""Main-app side of the speaker service: word alignment (DiartAttribution) and the streaming client.
Constructed service events and an in-memory connection: these test the integration logic, not audio accuracy."""
import json
import os
import queue
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from familiar.diart_align import DiartAttribution  # noqa: E402
from familiar.resolver import Resolver  # noqa: E402
from familiar.speaker_client import SpeakerClient  # noqa: E402
from familiar.voices import ME, VoicePrints  # noqa: E402


def line(book, speaker, words, status="provider"):
    book.feed(speaker, " ".join(w[0] for w in words), meta={"start": words[0][1], "end": words[-1][2],
                                                             "words": words, "status": status, "label": "1:%s" % speaker})


def seg(a, b, speakers, overlap=False, unidentified=False):
    return {"start": a, "end": b, "speakers": speakers, "overlap": overlap, "unidentified": unidentified}


def speaker_event(a, b, segments, gen=1, revision=0):
    return {"type": "speaker", "start": a, "end": b, "segments": segments, "generation": gen, "revision": revision,
            "attribution": "available", "processing_lag": 0.2}


def revision(label, wearer, gen=1, clean=5.0, applies_from=0.0, rev=1):
    return {"type": "revision", "speaker": label, "wearer": wearer, "previous": None, "generation": gen,
            "clean_seconds": clean, "applies_from": applies_from, "revision": rev}


class Alignment(unittest.TestCase):
    def setUp(self):
        self.book = Resolver()
        self.vp = VoicePrints(self.book, enabled=False)
        self.al = DiartAttribution(self.book, self.vp, apply=True)
        # Deepgram lumped everything under one label (person 3), as on 14 Nov
        line(self.book, 3, [("So", 0.2, 0.4), ("what", 0.4, 0.6), ("do", 0.6, 0.8), ("you", 0.8, 1.0), ("build?", 1.0, 1.4)])
        line(self.book, 3, [("Robot", 1.6, 1.9), ("arms", 1.9, 2.2), ("mostly", 2.2, 2.6), ("for", 2.6, 2.8), ("labs.", 2.8, 3.2)])

    def lines(self):
        return [(l["speaker"], l["text"], l.get("status")) for l in self.book.snapshot()["log"]]

    def test_wearer_and_other_words_are_moved(self):
        self.al.on_event(revision("s0", "wearer"))
        self.al.on_event(revision("s1", "other", rev=2))
        self.al.on_event(speaker_event(0.0, 3.5, [seg(0.0, 1.5, ["s0"]), seg(1.5, 3.5, ["s1"])]))
        got = self.lines()
        self.assertEqual(got[0][:2], (ME, "So what do you build?"))
        self.assertEqual(got[1][0], 3)                       # mapped to the existing person (anchors), not a new one
        self.assertTrue(self.book.people[ME]["is_me"])
        self.assertEqual(got[0][2], "corrected")

    def test_overlapping_word_stays_unresolved_with_candidates(self):
        self.al.on_event(revision("s0", "wearer"))
        self.al.on_event(revision("s1", "other", rev=2))
        self.al.on_event(speaker_event(0.0, 3.5, [seg(0.0, 1.5, ["s0"]), seg(1.5, 2.2, ["s0", "s1"], overlap=True),
                                                  seg(2.2, 3.5, ["s1"])]))
        robot = next(l for l in self.book.snapshot()["log"] if l["text"].startswith("Robot"))
        self.assertIsNone(robot["speaker"])
        self.assertIn(ME, robot["candidates"])

    def test_unidentified_overlap_is_ambiguous(self):
        self.al.on_event(revision("s0", "wearer"))
        self.al.on_event(speaker_event(0.0, 3.5, [seg(0.0, 1.5, ["s0"]), seg(1.5, 3.5, ["s0"], overlap=True,
                                                                                unidentified=True)]))
        robot = next(l for l in self.book.snapshot()["log"] if l["text"].startswith("Robot"))
        self.assertIsNone(robot["speaker"])                  # never "wearer" when another voice is also there

    def test_pending_speaker_changes_nothing_until_revised(self):
        self.al.on_event(speaker_event(0.0, 3.5, [seg(0.0, 1.5, ["s0"]), seg(1.5, 3.5, ["s1"])]))
        self.assertEqual([l[0] for l in self.lines()], [3, 3])
        self.al.on_event(revision("s0", "wearer", applies_from=0.0))      # later clean speech: s0 is the wearer
        self.assertEqual(self.lines()[0][0], ME)                           # earlier words are improved

    def test_weak_wearer_match_never_makes_a_person(self):
        self.al.on_event(revision("s1", "uncertain"))
        self.al.on_event(speaker_event(0.0, 3.5, [seg(0.0, 3.5, ["s1"])]))
        self.assertEqual(sorted(self.book.people), [3])

    def test_uncertain_speaker_words_leave_you(self):
        line(self.book, ME, [("Totally", 3.6, 4.0), ("agree", 4.0, 4.5)])
        self.al.on_event(revision("s2", "uncertain"))
        self.al.on_event(speaker_event(3.5, 5.0, [seg(3.5, 5.0, ["s2"])]))
        last = self.book.snapshot()["log"][-1]
        self.assertIsNone(last["speaker"])

    def test_generations_are_separate(self):
        self.al.on_event(revision("s0", "wearer", gen=1))
        self.al.on_event(speaker_event(0.0, 1.5, [seg(0.0, 1.5, ["s0"])], gen=2))   # "s0" of a NEW generation
        self.assertEqual(self.lines()[0][0], 3)              # not assumed to be the wearer

    def test_corrections_are_traceable_and_reversible(self):
        self.al.on_event(revision("s0", "wearer"))
        self.al.on_event(speaker_event(0.0, 1.5, [seg(0.0, 1.5, ["s0"])]))
        corr = self.book.corrections[-1]
        self.assertEqual(corr["engine"], "diart")
        self.assertEqual(corr["original"]["speaker"], 3)
        self.assertTrue(self.book.revert_correction(corr["id"]))
        self.assertEqual(self.lines()[0][0], 3)

    def test_status_reaches_the_page(self):
        self.al.on_event({"type": "status", "attribution": "unavailable", "reason": "behind", "generation": 1})
        self.assertEqual(self.book.snapshot()["attribution"], "unavailable")


class FakeSocket:
    def __init__(self, ready=True):
        self.sent, self.inbox, self.closed = [], queue.Queue(), False
        if ready:
            self.inbox.put(json.dumps({"type": "ready", "generation": 1}))

    def send(self, message):
        if self.closed:
            raise OSError("closed")
        self.sent.append(message)

    def recv(self, timeout=None):
        try:
            return self.inbox.get(timeout=timeout or 5)
        except queue.Empty:
            return None

    def close(self):
        self.closed = True
        self.inbox.put(None)


class Client(unittest.TestCase):
    def wait(self, cond, seconds=3.0):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if cond():
                return True
            time.sleep(0.02)
        return False

    def test_start_message_and_audio(self):
        sock = FakeSocket()
        events = []
        c = SpeakerClient("ws://x", "s1", events.append, connect=lambda: sock).start()
        c.feed(b"\x00\x00" * 320)
        self.assertTrue(self.wait(lambda: any(isinstance(m, bytes) for m in sock.sent)))
        start = json.loads(sock.sent[0])
        self.assertEqual((start["type"], start["sample_rate"], start["encoding"], start["start"]), ("start", 16000, "pcm_s16le", 0.0))
        self.assertEqual(events[0]["attribution"], "available")
        c.stop()

    def test_bounded_queue_drops_marks_gap_and_unavailable(self):
        events = []
        never = threading.Event()
        c = SpeakerClient("ws://x", "s2", events.append, max_buffer=1.0,
                          connect=lambda: (never.wait(10), (_ for _ in ()).throw(OSError("down")))[1])
        for _ in range(100):                                  # 2 s of audio while the service is away
            c.feed(b"\x00\x00" * 320)
        self.assertLessEqual(c._queued, 16000)
        self.assertGreater(c.stats["dropped_seconds"], 0.9)
        self.assertEqual(events[-1]["attribution"], "unavailable")
        sock = FakeSocket()
        c._connect = lambda: sock
        c.start()
        self.assertTrue(self.wait(lambda: any(isinstance(m, bytes) for m in sock.sent)))
        self.assertAlmostEqual(json.loads(sock.sent[0])["start"], 1.0, places=2)   # resumes where audio still is
        c.stop()

    def test_unavailable_service_is_reported(self):
        events = []
        c = SpeakerClient("ws://x", "s3", events.append,
                          connect=lambda: FakeSocket(ready=False) if not events else (_ for _ in ()).throw(OSError("x")))
        sock = FakeSocket(ready=False)
        sock.inbox.put(json.dumps({"type": "unavailable", "reason": "HF_TOKEN is not set"}))
        c._connect = lambda: sock
        c.start()
        self.assertTrue(self.wait(lambda: events))
        self.assertEqual(events[0]["attribution"], "unavailable")
        self.assertIn("HF_TOKEN", events[0]["reason"])
        c.stop()


if __name__ == "__main__":
    unittest.main()
