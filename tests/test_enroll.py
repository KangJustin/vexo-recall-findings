"""The wearer's voice baseline: enrolment checks, the profile, the voice test's numbers, and live use."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from familiar import enroll, voices  # noqa: E402
from familiar.resolver import Resolver  # noqa: E402
from test_voices import Conversation, fake_encoder, pcm, person, timed, tone  # noqa: E402

SR = enroll.SR


def speechy(freq, seconds, amp=0.2):
    """A tone in syllable-like bursts with short pauses, over faint noise: looks like speech to the detector."""
    t = np.arange(int(seconds * SR)) / SR
    bursts = (np.sin(2 * np.pi * 2.5 * t) > -0.3).astype(np.float32)
    noise = np.random.default_rng(0).standard_normal(len(t)).astype(np.float32) * 0.001
    return amp * np.sin(2 * np.pi * freq * t).astype(np.float32) * bursts + noise


class Samples(unittest.TestCase):
    def test_good_sample(self):
        speech, problem = enroll.check_sample(speechy(440, 4.0))
        self.assertIsNone(problem)
        self.assertGreater(len(speech) / SR, 2.0)

    def test_silence_is_refused(self):
        _, problem = enroll.check_sample(np.zeros(4 * SR, dtype=np.float32))
        self.assertIn("speech", problem)

    def test_too_short(self):
        _, problem = enroll.check_sample(np.concatenate([speechy(440, 1.0), np.zeros(3 * SR, np.float32)]))
        self.assertIn("only", problem)

    def test_clipping_is_refused(self):
        _, problem = enroll.check_sample(np.clip(speechy(440, 4.0, amp=3.0), -1, 1))
        self.assertIn("too loud", problem)

    def test_noisy_room_is_refused(self):
        noise = np.random.default_rng(1).standard_normal(4 * SR).astype(np.float32) * 0.1
        _, problem = enroll.check_sample(speechy(440, 4.0) + noise)
        self.assertIn("noise", problem)


class Profile(unittest.TestCase):
    def test_build_save_load_forget(self):
        samples = [enroll.check_sample(speechy(440, 4.0, amp=a))[0] for a in (0.2, 0.25, 0.3)]
        profile = enroll.build_profile(samples, fake_encoder)
        self.assertEqual(profile["samples"], 3)
        self.assertGreater(profile["consistency"], 0.9)
        self.assertTrue(0.30 <= profile["threshold"] <= 0.60)
        self.assertEqual(len(profile["embedding"]), 32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "me.json")
            enroll.save_profile(profile, path)
            loaded = enroll.load_profile(path)
            self.assertAlmostEqual(float(np.linalg.norm(loaded["vector"])), 1.0, places=4)
            self.assertTrue(enroll.forget_profile(path))
            self.assertIsNone(enroll.load_profile(path))
            self.assertFalse(enroll.forget_profile(path))

    def test_separation_numbers(self):
        s = enroll.separation([0.8, 0.7, 0.75, 0.6], [0.1, 0.2, 0.65, 0.3])
        self.assertEqual(s["accuracy"], 0.875)                   # one "other" (0.65) above the best cut
        self.assertEqual(s["you"]["min"], 0.6)
        self.assertEqual(s["others"]["max"], 0.65)
        self.assertEqual(enroll.at_threshold([0.8, 0.5], [0.1, 0.6], 0.55),
                         {"threshold": 0.55, "miss": 0.5, "false_accept": 0.5, "accuracy": 0.5})

    def test_report_on_two_voices(self):
        samples = [enroll.check_sample(speechy(440, 4.0))[0] for _ in range(3)]
        profile = enroll.build_profile(samples, fake_encoder)
        profile["vector"] = np.asarray(profile["embedding"], np.float32)
        you = enroll.speech_only(speechy(440, 10.0), margin=6.0)[0]
        others = enroll.speech_only(speechy(1800, 10.0, amp=0.05), margin=6.0)[0]
        out = enroll.report(you, others, profile, fake_encoder)
        self.assertEqual(out["windows"]["2s"]["separation"]["accuracy"], 1.0)
        self.assertGreater(out["loudness_db"]["you"], out["loudness_db"]["others"])


def me_profile(freq=440):
    v = fake_encoder(tone(freq, 2.0))
    return {"vector": v, "threshold": 0.6, "seconds": 12.0, "created": "test"}


class Live(unittest.TestCase):
    def conversation(self):
        c = Conversation()
        c.vp = voices.VoicePrints(c.book, encoder=fake_encoder, log=lambda _: None, me=me_profile())
        c.vp.ready.set()
        return c

    def test_wearer_is_you_from_the_first_line(self):
        c = self.conversation()
        c.say(1, 440, seconds=1.5, text="hey welcome")               # Deepgram calls the wearer "1"
        c.say(0, 1800, seconds=3.0, text="thanks, glad to be here")
        self.assertEqual(c.speakers(), [voices.ME, 1])
        self.assertTrue(person(c.book, voices.ME)["is_me"])
        self.assertEqual(person(c.book, voices.ME)["display"], "You")
        self.assertFalse(person(c.book, 1)["is_me"])

    def test_baseline_is_never_changed_by_live_audio(self):
        c = self.conversation()
        before = c.vp.voices[voices.ME]["sum"].copy()
        for _ in range(3):
            c.say(0, 440, seconds=3.0)
        np.testing.assert_array_equal(c.vp.voices[voices.ME]["sum"], before)

    def test_label_first_heard_on_a_short_line_is_claimed_later(self):
        c = self.conversation()
        c.say(3, 440, seconds=0.5, text="hi")                        # too short to check: a provisional person
        c.say(5, 1800, seconds=3.0, text="hello there, nice to meet you")
        c.say(3, 440, seconds=3.0, text="so what brings you to the meetup")
        self.assertEqual(c.speakers(), [voices.ME, 2, voices.ME])     # the provisional person was merged into You
        self.assertEqual({p["id"] for p in c.book.snapshot()["people"]}, {voices.ME, 2})

    def test_someone_else_is_not_you(self):
        c = self.conversation()
        c.say(0, 1800, seconds=3.0, text="hi I'm Maya")
        self.assertFalse(any(p["is_me"] for p in c.book.snapshot()["people"]))
        self.assertEqual(person(c.book, c.speakers()[0])["name"], "Maya")

    def test_without_baseline_loudness_still_decides(self):
        c = Conversation()
        for _ in range(2):
            c.say(0, 440, amp=0.5)
            c.say(1, 1800, amp=0.05)
        self.assertTrue(person(c.book, 0)["is_me"])


if __name__ == "__main__":
    unittest.main()
