"""The Deepgram reader, fed recorded-style messages (no network, no key). Skipped without the kit."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
KIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vexo-open-kit")

if os.path.isdir(KIT):
    from familiar.live import DiarizedDeepgram
    from familiar.resolver import Resolver


def results(words, final=False, speech_final=False):
    return {"type": "Results", "is_final": final, "speech_final": speech_final,
            "channel": {"alternatives": [{"transcript": " ".join(w for _, w in words),
                                          "words": [{"punctuated_word": w, "speaker": s} for s, w in words]}]}}


@unittest.skipUnless(os.path.isdir(KIT), "vexo-open-kit/ not cloned next to run.py")
class Live(unittest.TestCase):
    def setUp(self):
        self.got = []
        self.ears = DiarizedDeepgram(on_piece=lambda s, t, f: self.got.append((s, t, f)))
        self.ears.finals = []

    def finals(self):
        return [(s, t) for s, t, f in self.got if f]

    def test_query_uses_current_diarization_switch(self):
        self.assertEqual(self.ears.query["diarize_model"], "latest")
        self.assertNotIn("diarize", self.ears.query)            # sending both is rejected by Deepgram

    def test_split_intro_becomes_one_line(self):
        self.ears._message(results([(1, "Hi,"), (1, "I'm")], final=True))
        self.ears._message(results([(1, "Maya,"), (1, "I"), (1, "work"), (1, "at"), (1, "Northline.")],
                                   final=True, speech_final=True))
        self.assertEqual(self.finals(), [(1, "Hi, I'm Maya, I work at Northline.")])
        r = Resolver()
        r.feed(0, "Welcome.")
        for s, t in self.finals():
            r.feed(s, t)
        self.assertEqual([p["name"] for p in r.snapshot()["people"]], [None, "Maya"])

    def test_guess_shows_held_text(self):
        self.ears._message(results([(1, "Hi,"), (1, "I'm")], final=True))
        self.ears._message(results([(1, "Ma")]))
        self.assertEqual(self.got[-1], (1, "Hi, I'm Ma", False))
        self.assertEqual(self.finals(), [])

    def test_voice_change_flushes(self):
        self.ears._message(results([(0, "What's"), (0, "your"), (0, "name?"), (1, "Priya.")], final=True))
        self.assertEqual(self.finals(), [(0, "What's your name?")])
        self.ears._message({"type": "UtteranceEnd"})
        self.assertEqual(self.finals(), [(0, "What's your name?"), (1, "Priya.")])

    def test_close_flushes(self):
        self.ears._message(results([(2, "Thanks!")], final=True))
        self.ears._sentence_ends()
        self.assertEqual(self.finals(), [(2, "Thanks!")])



FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "deepgram_live")


def word(text, speaker, start, end):
    return {"word": text.lower().strip(".,?!"), "punctuated_word": text, "speaker": speaker, "start": start,
            "end": end, "confidence": 0.99}


def message(words, final=True, speech_final=True):
    return {"type": "Results", "is_final": final, "speech_final": speech_final,
            "channel": {"alternatives": [{"transcript": " ".join(w["punctuated_word"] for w in words), "words": words}]}}


@unittest.skipUnless(os.path.isdir(KIT), "vexo-open-kit/ not cloned next to run.py")
class Sentences(unittest.TestCase):
    """The new hand-off: settled lines with word timings and a connection id; provisional text replaced."""

    def setUp(self):
        self.lines, self.shown = [], []
        self.ears = DiarizedDeepgram(on_sentence=lambda *a: self.lines.append(a),
                                     on_interim=lambda pieces, conn: self.shown.append(list(pieces)))
        self.ears.finals = []

    def test_several_speakers_in_one_response_with_a_short_interruption(self):
        self.ears._message(message([word("The", 0, 1.0, 1.2), word("seam", 0, 1.2, 1.5), word("wait,", 1, 1.55, 1.8),
                                    word("which", 1, 1.8, 2.0), word("is", 0, 2.05, 2.2), word("sanded.", 0, 2.2, 2.6)]))
        got = [(a[0], a[1]) for a in self.lines]
        self.assertEqual(got, [(0, "The seam"), (1, "wait, which"), (0, "is sanded.")])   # "wait" kept, not merged
        self.assertEqual(self.lines[1][2:4], (1.55, 2.0))                                   # its own timing
        self.assertEqual(self.lines[1][4], [("wait,", 1.55, 1.8), ("which", 1.8, 2.0)])

    def test_interim_results_are_replaced_not_appended(self):
        self.ears._message(message([word("so", 0, 0.1, 0.3)], final=False))
        self.ears._message(message([word("so", 0, 0.1, 0.3), word("I", 0, 0.3, 0.4), word("was", 0, 0.4, 0.6)], final=False))
        self.assertEqual(self.shown[-1], [(0, "so I was")])
        self.ears._message(message([word("so", 0, 0.1, 0.3), word("I", 0, 0.3, 0.4), word("was", 0, 0.4, 0.6),
                                    word("saying.", 0, 0.6, 0.9)], final=True, speech_final=True))
        self.assertEqual([a[1] for a in self.lines], ["so I was saying."])                  # once, no duplicates
        self.assertEqual(self.shown[-1], [])                                                 # nothing provisional left

    def test_interim_with_two_speakers_shows_both(self):
        self.ears._message(message([word("Okay", 0, 0.1, 0.3), word("yeah", 1, 0.35, 0.5)], final=False))
        self.assertEqual(self.shown[-1], [(0, "Okay"), (1, "yeah")])

    def test_finalized_but_unfinished_text_stays_visible(self):
        self.ears._message(message([word("Hi,", 1, 0.1, 0.3), word("I'm", 1, 0.3, 0.5)], final=True, speech_final=False))
        self.ears._message(message([word("Ma", 1, 0.5, 0.7)], final=False))
        self.assertEqual(self.shown[-1], [(1, "Hi, I'm Ma")])
        self.assertEqual(self.lines, [])

    def test_each_connection_has_its_own_id(self):
        from unittest import mock
        import speech
        with mock.patch.object(speech.DeepgramSpeech, "start", lambda self, *a, **k: self):
            a = DiarizedDeepgram(on_sentence=lambda *x: None)
            b = DiarizedDeepgram(on_sentence=lambda *x: None)
            a.start()
            b.start()
        self.assertNotEqual(a.connection, b.connection)
        a.finals = []
        got = []
        a.on_sentence = lambda *x: got.append(x)
        a._message(message([word("Hello.", 0, 0.0, 0.4)]))
        self.assertEqual(got[0][5], a.connection)


@unittest.skipUnless(os.path.isdir(KIT) and os.path.isdir(FIXTURES), "needs the kit and recorded fixtures")
class RecordedPayloads(unittest.TestCase):
    """Real Deepgram live messages (eval/deepgram_live.py, 14 Nov 2026, synthetic voices), parsed by the app."""

    def replay(self, name):
        lines = []
        ears = DiarizedDeepgram(on_sentence=lambda *a: lines.append(a), on_interim=lambda *a: None)
        ears.finals = []
        msgs = [json.loads(l)["message"] for l in open(os.path.join(FIXTURES, name + ".jsonl"))]
        for m in msgs:
            ears._message(m)
        ears._sentence_ends()
        return lines, msgs

    def test_alternating_turns(self):
        lines, _ = self.replay("alternating")
        self.assertEqual([a[0] for a in lines], [0, 1, 0, 1])
        self.assertTrue(all(a[4] for a in lines))                                           # word timings kept

    def test_no_silence_provider_gives_one_label(self):
        """Deepgram's live diarizer put two people under one label here: a provider error the parser keeps."""
        lines, _ = self.replay("no_silence")
        self.assertEqual({a[0] for a in lines}, {0})

    def test_overlapped_interjection_is_missing_from_provider_output(self):
        _, msgs = self.replay("short_interruption")
        heard = " ".join(m["channel"]["alternatives"][0]["transcript"].lower() for m in msgs
                         if m.get("type") == "Results" and m.get("is_final"))
        self.assertNotIn("wait", heard)                                                     # never transcribed

    def test_streaming_words_carry_no_speaker_confidence(self):
        _, msgs = self.replay("backchannel")
        words = [w for m in msgs if m.get("type") == "Results" for w in m["channel"]["alternatives"][0].get("words", [])]
        self.assertTrue(words)
        self.assertFalse(any("speaker_confidence" in w for w in words))
        self.assertTrue(all("speaker" in w for w in words))

if __name__ == "__main__":
    unittest.main()
