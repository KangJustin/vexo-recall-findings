"""Both environments over a real WebSocket: the speaker service (its own venv, STAND-IN models) and the app's
client + word alignment. Checks the plumbing end to end; NOT evidence of accuracy on real voices."""
import json
import os
import subprocess
import sys
import time
import unittest
import urllib.request

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV = os.path.join(REPO, ".venv-speaker", "bin", "python")
sys.path.insert(0, REPO)


@unittest.skipUnless(os.path.exists(VENV), "speaker service environment not set up (.venv-speaker)")
class EndToEnd(unittest.TestCase):
    PORT = 8779

    @classmethod
    def setUpClass(cls):
        env = dict(os.environ, SPEAKER_TEST_STANDIN_MODELS="1", SPEAKER_LATENCY="1.0", SPEAKER_WEARER_MIN_SECONDS="1.0",
                   SPEAKER_WEARER_YES="0.8", SPEAKER_WEARER_NO="0.3",
                   SPEAKER_PROFILE=os.path.join(REPO, "speaker_service", "tests", "standin_profile.json"))
        cls.proc = subprocess.Popen([VENV, "-m", "uvicorn", "speaker_service.app:app", "--port", str(cls.PORT)],
                                    cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            try:
                if json.loads(urllib.request.urlopen("http://127.0.0.1:%d/ready" % cls.PORT, timeout=1).read())["ready"]:
                    return
            except Exception:
                time.sleep(0.25)
        raise RuntimeError("service did not become ready")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(5)

    def test_stream_align_and_correct(self):
        from familiar.diart_align import DiartAttribution
        from familiar.resolver import Resolver
        from familiar.speaker_client import SpeakerClient
        from familiar.voices import ME, VoicePrints
        sr = 16000
        tone = lambda f, s: (0.2 * np.sin(2 * np.pi * f * np.arange(int(s * sr)) / sr)).astype(np.float32)
        audio = np.concatenate([tone(440, 3), tone(1800, 3), np.zeros(3 * sr, np.float32)])   # wearer, then other
        book = Resolver()
        # Deepgram's words, all under one label (the provider error seen on 14 Nov)
        words = [("w%d" % i, 0.1 + i * 0.5, 0.5 + i * 0.5) for i in range(11)]
        book.feed(7, " ".join(w[0] for w in words), meta={"start": 0.1, "end": 5.5, "words": words, "status": "provider"})
        align = DiartAttribution(book, VoicePrints(book, enabled=False))
        client = SpeakerClient("ws://127.0.0.1:%d" % self.PORT, "e2e", align.on_event).start()
        pcm = (audio * 32767).astype("<i2").tobytes()
        for i in range(0, len(pcm), 640):                    # 20 ms packets, faster than real time
            client.feed(pcm[i:i + 640])
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and align.done_until.get(align.generation, 0) < 6.0:
            time.sleep(0.1)
        client.stop()
        log = book.snapshot()["log"]
        self.assertGreaterEqual(align.done_until.get(align.generation, 0), 6.0)
        self.assertTrue(align.stats["events"] > 5)
        speakers = [l["speaker"] for l in log]
        self.assertIn(7, speakers)                            # the other voice keeps the provider's person
        # no wearer profile in this test: nobody may be turned into "You"
        self.assertNotIn(ME, speakers)
        self.assertEqual(book.snapshot()["attribution"], "available")


if __name__ == "__main__":
    unittest.main()
