"""The real-time voice tracker, the turn-test scoring and the voice page's endpoint (fake voiceprints: tones)."""
import json
import os
import sys
import threading
import unittest
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from familiar import tracker as T  # noqa: E402
from familiar.band_voice import score_turns  # noqa: E402
from test_voices import fake_encoder, tone  # noqa: E402

SR = T.SR


def speech(freq, seconds, amp=0.2):
    """Tone bursts with short gaps over faint noise: speech-like for the detector."""
    t = np.arange(int(seconds * SR)) / SR
    bursts = (np.sin(2 * np.pi * 3 * t) > -0.6).astype(np.float32)
    noise = np.random.default_rng(int(freq)).standard_normal(len(t)).astype(np.float32) * 0.0005
    return amp * np.sin(2 * np.pi * freq * t).astype(np.float32) * bursts + noise


def me(freq=440, threshold=0.6, tracker_threshold=None):
    p = {"vector": fake_encoder(tone(freq, 1.0)), "threshold": threshold}
    if tracker_threshold is not None:
        p["tracker_threshold"] = tracker_threshold
    return p


def play(tr, clock, parts):
    """Feed [(audio, label)] in HOP steps; returns [(time, shown, truth)]."""
    out = []
    for audio, label in parts:
        hop = int(tr.hop * SR)
        for i in range(0, len(audio) - hop + 1, hop):
            tr.feed(audio[i:i + hop])
            clock[0] += tr.hop
            out.append((round(clock[0], 2), tr.step()["speaker"], label))
    return out


def fresh(**kw):
    clock = [0.0]
    return T.VoiceTracker(fake_encoder, clock=lambda: clock[0], **kw), clock


class Tracking(unittest.TestCase):
    def test_you_then_someone_new_then_you(self):
        tr, clock = fresh(me=me())
        got = play(tr, clock, [(speech(440, 3.0), "you"), (speech(1800, 3.0, 0.05), "other"), (speech(440, 3.0), "you")])
        self.assertEqual(got[11][1], "you")                                  # end of the first turn
        self.assertEqual(got[23][1], "p1")                                   # a new person learned in the second
        self.assertEqual(got[-1][1], "you")
        late = [s for t, s, truth in got if truth == "other" and t > 3.0 + 1.5]
        self.assertTrue(all(s == "p1" for s in late), late)

    def test_returning_person_is_recognised(self):
        tr, clock = fresh(me=me())
        play(tr, clock, [(speech(1800, 3.0, 0.05), "a"), (speech(440, 2.0), "you")])
        got = play(tr, clock, [(speech(1800, 2.0, 0.05), "a")])
        self.assertEqual(got[-1][1], "p1")
        self.assertEqual(tr.last["people"], ["p1"])                          # not a second new person

    def test_two_other_people(self):
        tr, clock = fresh(me=me())
        play(tr, clock, [(speech(1800, 3.0, 0.05), "a"), (speech(900, 4.0, 0.05), "b")])   # weakly distinct: ~2.25 s
        self.assertEqual(tr.last["people"], ["p1", "p2"])
        self.assertEqual(tr.current, "p2")

    def test_silence(self):
        tr, clock = fresh(me=me())
        play(tr, clock, [(speech(440, 2.0), "you")])
        got = play(tr, clock, [(np.random.default_rng(3).standard_normal(2 * SR).astype(np.float32) * 0.0005, "quiet")])
        self.assertEqual(got[-1][1], "silence")

    def test_steady_voice_does_not_flicker(self):
        tr, clock = fresh(me=me())
        got = play(tr, clock, [(speech(440, 6.0), "you")])
        shown = [s for t, s, _ in got if t > 1.5]
        self.assertEqual(set(shown), {"you"})

    def test_measured_tracker_threshold_wins(self):
        tr, _ = fresh(me=me(threshold=0.6, tracker_threshold=0.33))
        self.assertEqual(tr.me["threshold"], 0.33)
        tr2, _ = fresh(me=me(threshold=0.6))
        self.assertAlmostEqual(tr2.me["threshold"], 0.6 - T.ME_MARGIN)

    def test_without_baseline_nobody_is_you(self):
        tr, clock = fresh()
        got = play(tr, clock, [(speech(440, 3.0), "a"), (speech(1800, 3.0, 0.05), "b")])
        self.assertNotIn("you", {s for _, s, _ in got})
        self.assertEqual(tr.last["people"], ["p1", "p2"])


class TurnScoring(unittest.TestCase):
    def test_score_turns(self):
        segments = [("you", 0.0, 4.0), ("quiet", 4.0, 6.0), ("other", 6.0, 10.0)]
        log = [(0.5, "silence", None), (1.5, "you", 0.6), (2.5, "you", 0.55), (3.5, "unclear", 0.3),
               (5.5, "silence", None), (6.5, "you", 0.4), (7.5, "p1", 0.1), (8.5, "p1", 0.05), (9.5, "p1", 0.12)]
        r = score_turns(log, segments)
        self.assertEqual((r["you"]["right"], r["you"]["unclear"], r["you"]["n"]), (2, 1, 3))
        self.assertEqual((r["other"]["right"], r["other"]["n"]), (3, 3))     # 6.5 is inside the grace second
        self.assertEqual(r["quiet"]["right"], 1)
        self.assertEqual(r["delays"], [1.5, 1.5])
        self.assertEqual(r["others_made"], 1)
        self.assertEqual(r["calibration"]["accuracy"], 1.0)


class VoicePage(unittest.TestCase):
    def test_voice_state_endpoint(self):
        from familiar.server import make_voice_server
        tr, clock = fresh(me=me())
        play(tr, clock, [(speech(440, 2.0), "you")])
        server = make_voice_server(tr, 0, started=0.0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        try:
            st = json.loads(urllib.request.urlopen("http://127.0.0.1:%d/voice-state" % port).read())
            page = urllib.request.urlopen("http://127.0.0.1:%d/voice" % port).read().decode()
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(st["speaker"], "you")
        self.assertTrue(st["has_baseline"])
        self.assertAlmostEqual(st["threshold"], 0.6 - T.ME_MARGIN)
        self.assertTrue(st["history"])
        self.assertIn("Vexo Voice", page)


if __name__ == "__main__":
    unittest.main()
