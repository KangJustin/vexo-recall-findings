"""Who said each line. Unit tests use a fake voiceprint (tones stand in for voices); the RealModel tests run the
real speaker model on macOS text-to-speech voices and are skipped when the model or `say` is not available."""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from familiar import voices  # noqa: E402
from familiar.resolver import Resolver  # noqa: E402

SR = voices.SAMPLE_RATE


def tone(freq, seconds, amp=0.1):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def pcm(samples):
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


def fake_encoder(samples):
    """Energy in 32 frequency bands: the same tone gives the same vector, different tones are far apart."""
    spectrum = np.abs(np.fft.rfft(samples))
    bands = np.array([b.sum() for b in np.array_split(spectrum, 32)], dtype=np.float32)
    return bands / (np.linalg.norm(bands) or 1)


def timed(text, start, seconds):
    tokens = text.split()
    step = seconds / len(tokens)
    return [(w, start + i * step, start + (i + 1) * step) for i, w in enumerate(tokens)]


class Conversation:
    """Plays settled lines through VoicePrints like live mode: sound in, then Deepgram's line with its label."""

    def __init__(self, enabled=True):
        self.book = Resolver()
        if enabled:
            self.vp = voices.VoicePrints(self.book, encoder=fake_encoder, log=lambda _: None)
            self.vp.ready.set()
        else:
            self.vp = voices.VoicePrints(self.book, enabled=False)
        self.t = 0.0

    def say(self, label, freq, seconds=3.0, amp=0.1, text="something said here", connection=1):
        start = self.t
        self.vp.add_audio(pcm(tone(freq, seconds, amp)))
        self.t += seconds
        self.vp.process(label, text, start, self.t, timed(text, start, seconds), connection)

    def say_together(self, label, parts, connection=1):
        """One Deepgram line that really holds several voices: [(freq, seconds, amp, text), ...]."""
        start, words = self.t, []
        for freq, seconds, amp, text in parts:
            self.vp.add_audio(pcm(tone(freq, seconds, amp)))
            words += timed(text, self.t, seconds)
            self.t += seconds
        self.vp.process(label, " ".join(p[3] for p in parts), start, self.t, words, connection)

    def lines(self):
        return [(line["speaker"], line["text"]) for line in self.book.snapshot()["log"]]

    def speakers(self):
        return [line["speaker"] for line in self.book.snapshot()["log"]]

    def ids(self):
        return [p["id"] for p in self.book.snapshot()["people"]]


def person(book, sid):
    return next(p for p in book.snapshot()["people"] if p["id"] == sid)


class Buffer(unittest.TestCase):
    def test_cut_matches_timing_and_wraps(self):
        vp = voices.VoicePrints(Resolver(), encoder=fake_encoder)
        second = np.arange(SR, dtype=np.float32) / SR / 2
        for _ in range(voices.BUFFER_SECONDS + 5):
            vp.add_audio(pcm(second))
        got = vp.cut(voices.BUFFER_SECONDS + 1.25, voices.BUFFER_SECONDS + 1.5)
        self.assertEqual(len(got), SR // 4)
        self.assertAlmostEqual(float(got[0]), 0.125, places=3)
        self.assertIsNone(vp.cut(1, 2))
        self.assertIsNone(vp.cut(voices.BUFFER_SECONDS + 5, voices.BUFFER_SECONDS + 6))

    def test_one_chunk_longer_than_buffer_keeps_the_clock(self):
        vp = voices.VoicePrints(Resolver(), encoder=fake_encoder)
        vp.add_audio(pcm(np.zeros((voices.BUFFER_SECONDS + 3) * SR, dtype=np.float32)))
        self.assertEqual(vp._written, (voices.BUFFER_SECONDS + 3) * SR)


class LabelsFirst(unittest.TestCase):
    def test_alternating_labels_are_two_people(self):
        c = Conversation()
        c.say(0, 440, amp=0.4)
        c.say(1, 1800)
        c.say(0, 440, amp=0.4)
        self.assertEqual(c.speakers(), [0, 1, 0])
        self.assertEqual({l["status"] for l in c.book.snapshot()["log"]}, {"provider"})

    def test_short_interruption_keeps_deepgrams_label(self):
        """The 14 Nov failure: a 0.3 s "wait" Deepgram labelled speaker 1 was moved by a noisy voiceprint.
        Here the clip even SOUNDS like speaker 0; the label still wins on a clip this short."""
        c = Conversation()
        c.say(0, 440, seconds=3.0, text="so the enclosure is printed in two halves")
        c.say(1, 1800, seconds=3.0, text="right and then you glue it")
        c.say(0, 440, seconds=1.5, text="and then we glue")
        c.say(1, 440, seconds=0.3, text="wait")
        c.say(0, 440, seconds=1.5, text="the seam")
        self.assertEqual(c.lines()[-3:], [(0, "and then we glue"), (1, "wait"), (0, "the seam")])

    def test_labels_are_scoped_to_their_connection(self):
        c = Conversation(enabled=False)
        c.say(0, 440, connection=1, text="first connection speaker zero")
        c.say(0, 1800, connection=2, text="new connection, also called zero")
        self.assertEqual(c.speakers(), [0, 1])                     # not the same person just because "0"

    def test_new_label_with_a_known_voice_is_that_person(self):
        c = Conversation()
        c.say(0, 440, seconds=4.0, connection=1)
        c.say(1, 1800, seconds=4.0, connection=1)
        c.say(0, 1800, seconds=2.0, connection=2)                  # reconnected; "0" now is the 1800 voice
        self.assertEqual(c.speakers(), [0, 1, 1])
        self.assertEqual(c.book.snapshot()["log"][-1]["status"], "voice")

    def test_long_line_unlike_its_label_is_unresolved_and_teaches_nothing(self):
        c = Conversation()
        c.say(0, 440, seconds=4.0)
        before = c.vp.voices[0]["seconds"]
        c.say(0, 1800, seconds=3.0, text="this is someone else under label zero")
        last = c.book.snapshot()["log"][-1]
        self.assertIsNone(last["speaker"])
        self.assertEqual((last["status"], last["candidates"]), ("uncertain", [0]))
        self.assertEqual(c.vp.voices[0]["seconds"], before)

    def test_unresolved_line_names_nobody(self):
        c = Conversation()
        c.say(0, 440, seconds=4.0)
        c.say(0, 1800, seconds=3.0, text="Hi, I'm Maya and I just got here")
        self.assertIsNone(person(c.book, 0)["name"])
        self.assertEqual(len(c.ids()), 1)

    def test_interim_never_makes_a_card_and_is_replaced(self):
        c = Conversation()
        c.say(0, 440)
        c.vp.interim([(0, "so I was"), (5, "wait")], connection=1)
        snap = c.book.snapshot()
        self.assertEqual(c.ids(), [0])
        self.assertEqual(snap["interim"], [{"speaker": 0, "text": "so I was"}, {"speaker": None, "text": "wait"}])
        c.vp.interim([(0, "so I was saying")], connection=1)
        self.assertEqual(c.book.snapshot()["interim"], [{"speaker": 0, "text": "so I was saying"}])

    def test_without_a_model_labels_are_used(self):
        c = Conversation(enabled=False)
        c.say(4, 440)
        c.say(5, 1800)
        c.say(4, 440)
        self.assertEqual(c.speakers(), [0, 1, 0])


class CuttingIn(unittest.TestCase):
    def test_line_with_two_voices_is_split_at_the_right_word(self):
        c = Conversation()
        c.say(0, 440, amp=0.4, text="so what do you work on these days")
        c.say(1, 1800, amp=0.05, text="I build gripper enclosures for robots")
        c.say_together(1, [(1800, 2.5, 0.05, "and the hard part is the"),
                           (440, 2.5, 0.4, "sorry to cut in, which plastic")])
        self.assertEqual(c.lines()[-2:], [(1, "and the hard part is the"), (0, "sorry to cut in, which plastic")])
        self.assertEqual({l["status"] for l in c.book.snapshot()["log"][-2:]}, {"voice"})

    def test_cut_in_before_anyone_is_known(self):
        c = Conversation()
        c.say_together(0, [(440, 2.5, 0.4, "hey are you here for the"), (1800, 2.5, 0.05, "yes I am, first time here")])
        self.assertEqual(c.speakers(), [0, 1])

    def test_one_voice_getting_louder_is_not_split(self):
        c = Conversation()
        c.say(0, 440, amp=0.2, text="Hello there everyone.")
        c.say_together(0, [(440, 2.5, 0.4, "and then I said look"), (440, 2.5, 0.05, "it really does not matter")])
        self.assertEqual(c.speakers(), [0, 0])

    def test_short_lines_are_never_split(self):
        c = Conversation()
        c.say_together(0, [(440, 0.8, 0.4, "hi there"), (1800, 0.8, 0.05, "hello you")])
        self.assertEqual(len(c.lines()), 1)


class Matching(unittest.TestCase):
    def test_loudest_voice_is_the_wearer(self):
        c = Conversation()
        for _ in range(2):
            c.say(0, 440, amp=0.5)
            c.say(1, 1800, amp=0.05)
        self.assertTrue(person(c.book, 0)["is_me"])
        self.assertFalse(person(c.book, 1)["is_me"])

    def test_similar_loudness_names_no_wearer(self):
        c = Conversation()
        for _ in range(2):
            c.say(0, 440, amp=0.2)
            c.say(1, 1800, amp=0.15)
        self.assertFalse(any(p["is_me"] for p in c.book.snapshot()["people"]))

    def test_new_conversation_forgets_voices(self):
        c = Conversation()
        c.say(1, 220)
        c.book.reset()
        c.say(5, 1800)
        self.assertEqual(len(c.vp.voices), 1)
        self.assertEqual(c.ids(), [0])

    def test_user_merge_is_followed(self):
        c = Conversation()
        c.say(0, 440)
        c.say(1, 1800)
        c.book.merge(1, 0)                                         # the user: "Same as..."
        c.say(1, 1800)
        self.assertEqual(c.ids(), [0])

    def test_worker_survives_a_failing_model(self):
        said = []
        book = Resolver()
        vp = voices.VoicePrints(book, loader=lambda: (_ for _ in ()).throw(ImportError("No module named torch")),
                                log=said.append).start()
        for _ in range(40):
            if vp.failed:
                break
            time.sleep(0.05)
        self.assertIn("Voice matching is off: No module named torch", said)
        vp.settle(2, "Hello.", 0.0, 1.0, [("Hello.", 0.0, 1.0)], 1)
        for _ in range(40):
            if book.snapshot()["log"]:
                break
            time.sleep(0.05)
        self.assertEqual(book.snapshot()["log"][0]["speaker"], 0)  # shown, as the label's person
        vp.stop()


def have_real_model():
    try:
        import speechbrain  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return shutil.which("say") is not None and os.path.isdir(voices.MODEL_DIR)


@unittest.skipUnless(have_real_model(), "needs torch, speechbrain, the downloaded model and macOS `say`")
class RealModel(unittest.TestCase):
    """Real ECAPA voiceprints on text-to-speech voices, fed through the live path (DiarizedDeepgram messages)."""

    @classmethod
    def setUpClass(cls):
        cls.encode = staticmethod(voices.load_encoder())
        cls.dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def speak(self, voice, text):
        path = os.path.join(self.dir, "%s-%d.wav" % (voice, abs(hash(text))))
        subprocess.run(["say", "-v", voice, "-o", path, "--data-format=LEI16@16000", text], check=True)
        with wave.open(path) as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768

    def run_script(self, script):
        kit = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vexo-open-kit")
        if not os.path.isdir(kit):
            self.skipTest("vexo-open-kit/ not cloned")
        from familiar.live import DiarizedDeepgram
        book = Resolver()
        vp = voices.VoicePrints(book, encoder=self.encode, log=lambda _: None)
        vp.ready.set()
        ears = DiarizedDeepgram(on_sentence=vp.settle, on_interim=vp.interim)
        ears.finals = []
        t = 0.0
        for item in script:
            dg_voice, parts = (item[0], [item[1:]]) if isinstance(item[1], str) else (item[0], item[1])
            words = []
            for mac_voice, gain, text in parts:
                sound = self.speak(mac_voice, text) * gain
                vp.add_audio(pcm(sound))
                tokens = text.split()
                step = len(sound) / SR / len(tokens)
                words += [{"punctuated_word": w, "speaker": dg_voice, "start": t + i * step,
                           "end": t + (i + 1) * step} for i, w in enumerate(tokens)]
                t += len(sound) / SR
            ears._message({"type": "Results", "is_final": True, "speech_final": True, "channel": {"alternatives": [{
                "transcript": " ".join(w["punctuated_word"] for w in words), "words": words}]}})
            while not vp._jobs.empty():
                vp.process(*vp._jobs.get())
        return book, vp

    def test_everyone_labelled_zero_on_separate_lines(self):
        """Deepgram lumps a second voice into label 0: those lines are left unresolved, never given to the
        first person."""
        book, vp = self.run_script([
            (0, "Samantha", 1.0, "Hey, welcome to the hardware meetup, are you here for the robot demos?"),
            (0, "Daniel", 0.2, "Yes, first time here. My name is Rafa and I work on gripper enclosures."),
            (0, "Samantha", 1.0, "Oh nice, we have been struggling with gripper housings for months now."),
            (0, "Daniel", 0.2, "The trick is a softer plastic on the inside and a stiff shell outside it."),
        ])
        self.assertEqual([line["speaker"] for line in book.snapshot()["log"]], [0, None, 0, None], vp.decisions)
        self.assertIsNone(person(book, 0)["name"])                 # Rafa's intro never named Samantha

    def test_cutting_in(self):
        book, vp = self.run_script([
            (0, "Samantha", 1.0, "Hey, welcome to the hardware meetup, are you here for the robot demos?"),
            (1, "Daniel", 0.2, "Yes, first time here. I work on gripper enclosures for small robots."),
            (1, [("Daniel", 0.2, "The trick is a softer plastic on the inside and"),
                 ("Samantha", 1.0, "Sorry to cut in, which plastic do you use for that?")]),
        ])
        lines = [(line["speaker"], line["text"]) for line in book.snapshot()["log"]]
        self.assertEqual(lines[-2:], [(1, "The trick is a softer plastic on the inside and"),
                                      (0, "Sorry to cut in, which plastic do you use for that?")], vp.decisions)

    def test_cutting_in_before_anyone_is_known(self):
        book, vp = self.run_script([
            (0, [("Samantha", 1.0, "Hey, welcome to the meetup, are you here for the"),
                 ("Daniel", 0.2, "Yes I am, it is my first time here and I love it")]),
        ])
        self.assertEqual([line["speaker"] for line in book.snapshot()["log"]], [0, 1], vp.decisions)

    def test_split_label_is_one_person_and_wearer(self):
        book, vp = self.run_script([
            (0, "Samantha", 1.0, "Hey, welcome to the hardware meetup, are you here for the robot demos?"),
            (1, "Daniel", 0.2, "Yes, first time here. I work on gripper enclosures for small warehouse robots."),
            (0, "Samantha", 1.0, "Oh nice, we have been struggling with gripper housings for months now."),
            (2, "Daniel", 0.2, "The trick is a softer plastic on the inside and a stiff shell outside it."),
            (0, "Samantha", 1.0, "That makes sense, thank you, I will tell the team about that one."),
        ])
        self.assertEqual([p["id"] for p in book.snapshot()["people"]], [0, 1], vp.decisions)
        self.assertTrue(person(book, 0)["is_me"])


if __name__ == "__main__":
    unittest.main()
