"""Speaker service MECHANICS with stand-in models (see fakes.py) and diart's real clustering/aggregation.

    .venv-speaker/bin/python -m unittest discover -s speaker_service/tests -t .

These check the timeline, events, wearer revisions, overlap flags, separate sessions, overload, reconnect gaps,
enrolment checks and the WebSocket protocol. They are NOT evidence of accuracy on real voices.
"""
import json
import os
import tempfile
import unittest

import numpy as np

from speaker_service import enrollment
from speaker_service.config import Settings
from speaker_service.engine import Models, Session, _access_message
from speaker_service.tests.fakes import VOICES, fake_models, profile_for, silence, tone

SR = 16000


_NO_PROFILE = os.path.join(tempfile.gettempdir(), "vexo-speaker-tests-no-profile.json")


def settings(**kw):
    s = Settings()
    s.profile = _NO_PROFILE                     # never the real enrolled wearer on this machine
    s.wearer_yes, s.wearer_no, s.wearer_min_seconds, s.max_lag, s.latency = 0.8, 0.3, 1.0, 30.0, 1.0
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def run(session, audio, chunk=0.02):
    n = int(chunk * SR)
    for i in range(0, len(audio), n):
        session.feed(audio[i:i + n])
        session.process()


def collect(models, s, audio, profile=None, chunk=0.02):
    events = []
    session = Session(models, "t", s, profile=profile, on_event=events.append)
    run(session, audio, chunk)
    return session, events


def at(events, t):
    """The speaker segment covering time t."""
    for e in events:
        if e["type"] == "speaker":
            for seg in e["segments"]:
                if seg["start"] <= t < seg["end"]:
                    return seg
    return None


class Models_(unittest.TestCase):
    def test_gated_model_without_token_says_what_to_do(self):
        msg = _access_message("pyannote/segmentation-3.0", Exception("401"), has_token=False)
        self.assertIn("HF_TOKEN is not set", msg)
        self.assertIn("huggingface.co/pyannote/segmentation-3.0", msg)

    def test_errors_never_contain_the_token(self):
        s = Settings()
        s.hf_token = "hf_secret_value_123"
        s.segmentation = "pyannote/does-not-exist-xyz"
        m = Models(s, embedding_loader=lambda: None)
        m._load_segmentation = lambda: (_ for _ in ()).throw(RuntimeError("bad " + s.hf_token))
        m.load()
        self.assertEqual(m.status, "error")
        self.assertNotIn("hf_secret_value_123", json.dumps(m.summary()))
        self.assertNotIn("hf_secret_value_123", json.dumps(s.public()))

    def test_window_comes_from_the_model(self):
        m = fake_models(settings(), duration=4.0)
        self.assertEqual((m.status, m.duration, m.config.duration), ("ready", 4.0, 4.0))


class Timeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = fake_models(settings())

    def test_events_cover_the_timeline_without_holes(self):
        audio = np.concatenate([tone(VOICES["A"], 3), tone(VOICES["B"], 3), silence(2)])
        session, events = collect(self.models, settings(), audio)
        spans = [(e["start"], e["end"]) for e in events if e["type"] == "speaker"]
        self.assertAlmostEqual(spans[0][0], 0.0)
        for (a, b), (c, d) in zip(spans, spans[1:]):
            self.assertAlmostEqual(b, c, places=3)                      # contiguous, no repeats
        self.assertGreater(spans[-1][1], 8 - session.latency - 0.6)    # up to the latency behind the audio
        self.assertTrue(all(e["algorithmic_latency"] == session.latency for e in events if e["type"] == "speaker"))

    def test_speaker_change_without_silence(self):
        audio = np.concatenate([tone(VOICES["A"], 3), tone(VOICES["B"], 3), silence(2)])
        _, events = collect(self.models, settings(), audio)
        a, b = at(events, 1.5)["speakers"], at(events, 4.5)["speakers"]
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 1)
        self.assertNotEqual(a, b)

    def test_overlap_and_no_speech(self):
        audio = np.concatenate([tone(VOICES["A"], 3), tone(VOICES["B"], 3), tone(VOICES["A"], 2) + tone(VOICES["B"], 2),
                                silence(3)])
        _, events = collect(self.models, settings(), audio)
        both = at(events, 6.5)                                  # early in the overlap: both identified
        self.assertTrue(both["overlap"])
        self.assertEqual(len(both["speakers"]), 2)
        deep = at(events, 7.6)                                  # later: one voice has no clean speech in the window,
        self.assertTrue(deep["overlap"])                        # diart cannot identify it, but the overlap is still
        self.assertTrue(deep.get("unidentified") or len(deep["speakers"]) == 2)   # flagged (raw segmentation)
        quiet = at(events, 9.0)
        self.assertEqual((quiet["speakers"], quiet["wearer"]), ([], "none"))

    def test_multiple_others_overlapping(self):
        audio = np.concatenate([tone(VOICES["B"], 3), tone(VOICES["C"], 3), tone(VOICES["B"], 2) + tone(VOICES["C"], 2),
                                silence(2)])
        _, events = collect(self.models, settings(), audio, profile=profile_for(VOICES["A"]))
        seg = at(events, 6.5)
        self.assertTrue(seg["overlap"])
        self.assertEqual(seg["wearer"], "other")
        deep = at(events, 7.6)                                  # an unidentified voice could be anyone: not "other"
        self.assertTrue(deep["overlap"])
        self.assertIn(deep["wearer"], ("other", "uncertain"))


class Wearer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = fake_models(settings())

    def test_wearer_and_other_are_told_apart_with_revisions(self):
        audio = np.concatenate([tone(VOICES["A"], 3), tone(VOICES["B"], 3), tone(VOICES["A"], 2) + tone(VOICES["B"], 2),
                                silence(2)])
        session, events = collect(self.models, settings(), audio, profile=profile_for(VOICES["A"]))
        self.assertEqual(at(events, 2.0)["wearer"], "wearer")
        self.assertEqual(at(events, 5.0)["wearer"], "other")
        self.assertEqual(at(events, 6.5)["wearer"], "wearer+other")
        self.assertIn(at(events, 7.6)["wearer"], ("wearer+other", "uncertain"))     # never plain "wearer" or "other"
        revisions = [e for e in events if e["type"] == "revision"]
        self.assertTrue(any(r["wearer"] == "wearer" for r in revisions))
        self.assertTrue(all(r["previous"] in (None, "pending") or r["previous"] != r["wearer"] for r in revisions))
        statuses = {k: v["wearer"] for k, v in session.speakers.items()}
        self.assertEqual(sorted(statuses.values()), ["other", "wearer"])

    def test_first_moments_are_pending_not_guessed(self):
        _, events = collect(self.models, settings(wearer_min_seconds=5.0), tone(VOICES["A"], 3),
                            profile=profile_for(VOICES["A"]))
        self.assertEqual(at(events, 0.5)["wearer"], "uncertain")          # pending: not called wearer yet

    def test_without_profile_nobody_is_the_wearer(self):
        _, events = collect(self.models, settings(), np.concatenate([tone(VOICES["A"], 3), silence(2)]))
        self.assertEqual(at(events, 1.0)["wearer"], "uncertain")


class State(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = fake_models(settings())

    def test_sessions_do_not_share_speaker_state(self):
        e1, e2 = [], []
        s1 = Session(self.models, "one", settings(), on_event=e1.append)
        s2 = Session(self.models, "two", settings(), on_event=e2.append)
        run(s1, np.concatenate([tone(VOICES["A"], 3), tone(VOICES["B"], 3)]))
        run(s2, tone(VOICES["C"], 3))
        self.assertEqual(len(s1.speakers), 2)
        self.assertEqual(len(s2.speakers), 1)
        self.assertIsNot(s1.pipeline.clustering, s2.pipeline.clustering)
        self.assertIs(s1.pipeline.config.segmentation, s2.pipeline.config.segmentation)   # weights shared

    def test_reconnect_gap_is_marked_and_timeline_continues(self):
        events = []
        s = Session(self.models, "g", settings(), on_event=events.append)
        run(s, tone(VOICES["A"], 2))
        s.feed(tone(VOICES["A"], 2), start=3.0)                           # one second never arrived
        s.process()
        gaps = [e for e in events if e["type"] == "gap"]
        self.assertEqual((gaps[0]["start"], gaps[0]["end"]), (2.0, 3.0))
        self.assertAlmostEqual(s.received, 5.0)

    def test_repeated_audio_after_reconnect_is_not_duplicated(self):
        s = Session(self.models, "d", settings(), on_event=lambda e: None)
        run(s, tone(VOICES["A"], 2))
        s.feed(tone(VOICES["A"], 1), start=1.5)                           # 0.5 s already received
        self.assertAlmostEqual(s.received, 2.5)

    def test_a_burst_of_audio_is_not_overload(self):
        events = []
        s = Session(self.models, "b", settings(max_lag=1.0), on_event=events.append)
        s.feed(tone(VOICES["A"], 6))                                       # 6 s at once, processed promptly
        s.process()
        self.assertNotIn("gap", [e["type"] for e in events])

    def test_overload_marks_attribution_unavailable_and_skips_ahead(self):
        events = []
        now = [100.0]
        s = Session(self.models, "o", settings(max_lag=1.0), on_event=events.append, clock=lambda: now[0])
        s.feed(tone(VOICES["A"], 6))
        now[0] += 3.0                                                      # processing could not run for 3 s
        s.process()
        kinds = [e["type"] for e in events]
        self.assertIn("status", kinds)
        self.assertEqual(next(e for e in events if e["type"] == "status")["attribution"], "unavailable")
        self.assertIn("gap", kinds)
        self.assertIn("generation", kinds)
        run(s, tone(VOICES["A"], 3))
        self.assertTrue(any(e["type"] == "speaker" for e in events))


class Enrolment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = fake_models(settings())

    def speechy(self, freq, seconds):
        t = np.arange(int(seconds * SR)) / SR
        return tone(freq, seconds) * (np.sin(2 * np.pi * 2.5 * t) > -0.3) + np.random.default_rng(0).standard_normal(len(t)).astype(np.float32) * 0.001

    def test_clean_clip_accepted_and_profile_records_models(self):
        clips = [enrollment.analyse_clip(self.speechy(VOICES["A"], 4), self.models) for _ in range(3)]
        self.assertTrue(all(c["ok"] for c in clips), [c["problem"] for c in clips])
        profile = enrollment.build_profile(clips, self.models, settings())
        self.assertEqual(profile["embedding_model"], "speechbrain/spkrec-ecapa-voxceleb")
        self.assertEqual(profile["preprocessing"]["sample_rate"], 16000)
        self.assertTrue(profile["confirmed"])
        with tempfile.TemporaryDirectory() as d:
            enrollment.save(profile, os.path.join(d, "p.json"))
            other = settings()
            other.embedding = "some/other-model"
            self.assertIsNone(enrollment.load(os.path.join(d, "p.json"), other))   # never compared across models

    def test_silence_and_overlapping_voices_refused(self):
        self.assertIn("speech", enrollment.analyse_clip(silence(4), self.models)["problem"])
        mixed = self.speechy(VOICES["A"], 4) + self.speechy(VOICES["B"], 4)
        self.assertIn("voice", enrollment.analyse_clip(mixed, self.models)["problem"])

    def test_decode_resamples(self):
        import io
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(48000)
            w.writeframes((np.tile(tone(440, 1.0, 0.3)[:48000, None], (1, 2)) * 32767).astype("<i2").tobytes())
        x = enrollment.decode(buf.getvalue())
        self.assertAlmostEqual(len(x) / SR, 1.0 / 3, places=2)            # 16000 samples of 48 kHz audio = 1/3 s


class Protocol(unittest.TestCase):
    def test_websocket_stream(self):
        from fastapi.testclient import TestClient
        import speaker_service.app as service
        service.models = fake_models(settings())
        service.settings.latency, service.settings.wearer_min_seconds = 1.0, 1.0
        service.settings.profile = _NO_PROFILE
        with TestClient(service.app) as client:
            self.assertEqual(client.get("/health").json(), {"ok": True})
            self.assertEqual(client.get("/ready").status_code, 200)
            with client.websocket_connect("/sessions/demo/audio") as ws:
                ws.send_text(json.dumps({"type": "start", "sample_rate": 16000, "encoding": "pcm_s16le", "channels": 1}))
                ready = json.loads(ws.receive_text())
                self.assertEqual(ready["type"], "ready")
                pcm = (np.concatenate([tone(VOICES["A"], 3), tone(VOICES["B"], 3)]) * 32767).astype("<i2").tobytes()
                for i in range(0, len(pcm), 640):
                    ws.send_bytes(pcm[i:i + 640])
                ws.send_text(json.dumps({"type": "end"}))
                events = []
                while True:
                    e = json.loads(ws.receive_text())
                    events.append(e)
                    if e["type"] == "summary":
                        break
            self.assertTrue(any(e["type"] == "speaker" for e in events))
            self.assertTrue(all(e["session_id"] == "demo" for e in events if e["type"] == "speaker"))
            self.assertIn("processing_lag_median", events[-1])

    def test_unready_models_are_reported_not_hidden(self):
        from fastapi.testclient import TestClient
        import speaker_service.app as service
        service.models = Models(settings())
        service.models.status, service.models.error = "error", "HF_TOKEN is not set"
        with TestClient(service.app) as client:
            client.app.router.on_startup.clear()
            with client.websocket_connect("/sessions/x/audio") as ws:
                ws.send_text(json.dumps({"type": "start", "sample_rate": 16000}))
                self.assertEqual(json.loads(ws.receive_text()), {"type": "unavailable", "reason": "HF_TOKEN is not set"})


if __name__ == "__main__":
    unittest.main()
