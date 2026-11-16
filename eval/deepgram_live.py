"""Stream each labelled scenario to Deepgram LIVE, exactly like the band does, and keep every raw message.

The audio is encoded as 20 ms Opus packets (the band's format) and sent in real time through the app's
own DiarizedDeepgram (familiar/live.py), so the query parameters are the app's. Every message Deepgram
sends back is saved with the wall-clock moment it arrived and how much audio had been sent by then.

    python3 eval/deepgram_live.py [scenario ...]    -> eval/results/deepgram_live/<scenario>.jsonl

Sends synthetic (text-to-speech) audio to Deepgram with the team's DEEPGRAM_API_KEY.
"""
import json
import os
import sys
import threading
import time

import av
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import corpus  # noqa: E402
from familiar.live import DiarizedDeepgram  # noqa: E402

OUT = os.path.join(HERE, "results", "deepgram_live")


def opus_packets(audio):
    """20 ms Opus packets at 16 kHz mono, like the band's."""
    enc = av.CodecContext.create("libopus", "w")
    enc.sample_rate = corpus.SR
    enc.layout = "mono"
    enc.format = "s16"
    enc.bit_rate = 24000
    enc.options = {"frame_duration": "20", "application": "voip"}
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    packets = []
    for i in range(0, len(pcm) - 319, 320):
        frame = av.AudioFrame.from_ndarray(pcm[i:i + 320][None, :], format="s16", layout="mono")
        frame.sample_rate = corpus.SR
        frame.pts = i
        packets += [bytes(p) for p in enc.encode(frame)]
    packets += [bytes(p) for p in enc.encode(None)]
    return packets


class Capture(DiarizedDeepgram):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.raw, self.t0, self.sent = [], None, 0.0

    def _message(self, message):
        self.raw.append({"received": round(time.monotonic() - self.t0, 3), "audio_sent": round(self.sent, 3),
                         "message": message})
        return super()._message(message)


def run(name):
    audio, truth = corpus.load(name)
    packets = opus_packets(audio)
    ears = Capture(on_piece=lambda *a: None)
    ears.t0 = time.monotonic()
    ears.start()
    start = time.monotonic()
    for i, packet in enumerate(packets):                     # real time: one 20 ms packet every 20 ms
        ears.feed(packet)
        ears.sent = (i + 1) * 0.02
        wait = start + (i + 1) * 0.02 - time.monotonic()
        if wait > 0:
            time.sleep(wait)
    ears.stop()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".jsonl"), "w") as f:
        for item in ears.raw:
            f.write(json.dumps(item) + "\n")
    finals = [m for m in ears.raw if m["message"].get("type") == "Results" and m["message"].get("is_final")]
    speakers = {w.get("speaker") for m in finals for w in m["message"]["channel"]["alternatives"][0].get("words", [])}
    print("%-18s %3d messages, %2d final, speakers %s, query diarize_model=%s" % (
        name, len(ears.raw), len(finals), sorted(s for s in speakers if s is not None), ears.query.get("diarize_model")))


def main():
    names = sys.argv[1:] or corpus.names()
    threads = [threading.Thread(target=run, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
