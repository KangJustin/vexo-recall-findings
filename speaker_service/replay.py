"""Replay a recording through the speaker service pipeline, score it, or calibrate the wearer thresholds.

    # a recorded Vexo conversation (run.py --record PATH saves PATH.wav + PATH.jsonl), all at once:
    .venv-speaker/bin/python -m speaker_service.replay eval/corpus/talk1.wav --truth eval/corpus/talk1.json --wearer A
    # the same, as live 20 ms chunks at real-time pace (measures steady-state processing lag):
    .venv-speaker/bin/python -m speaker_service.replay eval/corpus/talk1.wav --truth ... --wearer A --stream --realtime
    # calibrate wearer thresholds on recordings that are NOT used for evaluation:
    .venv-speaker/bin/python -m speaker_service.replay --calibrate --wearer-clips me1.wav me2.wav --other-clips them1.wav

Truth files use the eval/ format: {"turns": [{"speaker", "start", "end", ...}], "overlaps": [[start, end], ...]};
--wearer names the truth speaker who wears the band. With --deepgram PATH.jsonl (the recording's Deepgram live
messages), words are scored too. Results print and save as JSON (scores and timings only, never audio).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import wave

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from speaker_service import enrollment  # noqa: E402
from speaker_service.config import Settings  # noqa: E402
from speaker_service.engine import Models, Session, Worker  # noqa: E402

CALIBRATION = os.path.join(REPO, "profiles", "calibration.json")
STEP = 0.1               # seconds between scored moments


def read_wav(path):
    with open(path, "rb") as f:
        return enrollment.decode(f.read())


def digest(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def load_models(settings, standin=False):
    started = time.monotonic()
    if standin:                                           # mechanics only: tones stand in for voices
        from speaker_service.tests.fakes import FakeEmbedding, FakeSegmentation
        models = Models(settings, segmentation_loader=lambda: FakeSegmentation(), embedding_loader=lambda: FakeEmbedding())
        models.specs = {"duration": 5.0}
        models.load()
    else:
        models = Models(settings).load()
    if not models.ready():
        sys.exit("Models not ready: %s" % models.error)
    return models, round(time.monotonic() - started, 2)


def run(models, settings, audio, profile, stream=False, realtime=False):
    """Events from one recording. stream: 20 ms chunks through a worker thread, as live."""
    events = []
    if not realtime:                  # faster than real time: audio "waits" by design, that is not overload
        settings.max_lag = float("inf")
    session = Session(models, "replay", settings, profile=profile, on_event=events.append)
    started = time.monotonic()
    if not stream:
        session.feed(audio)
        session.process()
    else:
        worker = Worker(session).start()
        n = int(0.02 * 16000)
        t0 = time.monotonic()
        for i in range(0, len(audio), n):
            session.feed(audio[i:i + n])
            if realtime:
                time.sleep(max(0.0, t0 + (i + n) / 16000 - time.monotonic()))
        while session.backlog() > session.step + 1e-6:
            time.sleep(0.05)
        worker.stop(flush=True)
    return events, session.summary(), round(time.monotonic() - started, 2)


# ---------------------------------------------------------------- scoring

def truth_at(truth, t):
    who = sorted({x["speaker"] for x in truth["turns"] if x["start"] <= t < x["end"]})
    overlap = len(who) > 1 or any(a <= t < b for a, b in truth.get("overlaps", []))
    return who, overlap


def predicted_at(events, t):
    for e in events:
        if e["type"] == "speaker" and e["start"] <= t < e["end"]:
            for seg in e["segments"]:
                if seg["start"] <= t < seg["end"]:
                    return seg
    return None


def score_moments(events, truth, wearer):
    """Every STEP seconds of the processed timeline: who really spoke vs what the service said."""
    end = max([e["end"] for e in events if e["type"] == "speaker"] or [0])
    c = {"moments": 0, "wearer_alone": 0, "other_alone": 0, "overlap": 0, "silence": 0,
         "other_as_wearer": 0, "wearer_as_other": 0, "wearer_right": 0, "other_right": 0, "unresolved": 0,
         "overlap_flagged": 0, "false_overlap": 0, "missed_speech": 0, "false_speech": 0}
    for t in np.arange(0.05, end, STEP):
        who, overlap = truth_at(truth, t)
        seg = predicted_at(events, t)
        if seg is None:
            continue
        c["moments"] += 1
        said = seg["wearer"]
        if overlap:
            c["overlap"] += 1
            c["overlap_flagged"] += bool(seg["overlap"])
            continue
        if seg["overlap"]:
            c["false_overlap"] += 1
        if not who:
            c["silence"] += 1
            if seg["speakers"] or seg.get("unidentified"):
                c["false_speech"] += 1
            continue
        if not seg["speakers"] and not seg.get("unidentified"):
            c["missed_speech"] += 1
        if who == [wearer]:
            c["wearer_alone"] += 1
            if said == "wearer":
                c["wearer_right"] += 1
            elif said == "other":
                c["wearer_as_other"] += 1
            else:
                c["unresolved"] += 1
        else:
            c["other_alone"] += 1
            if said in ("wearer", "wearer+other"):
                c["other_as_wearer"] += 1
            elif said == "other":
                c["other_right"] += 1
            else:
                c["unresolved"] += 1
    return c


def deepgram_words(path):
    words = []
    for line in open(path):
        m = json.loads(line)["message"]
        if m.get("type") == "Results" and m.get("is_final"):
            for w in m["channel"]["alternatives"][0].get("words", []):
                words.append((w.get("punctuated_word") or w["word"], w["start"], w["end"]))
    return words


def score_words(events, truth, wearer, words):
    """Words as the app would attribute them (familiar/diart_align.py rules) vs who said them."""
    from familiar.diart_align import DiartAttribution
    from familiar.resolver import Resolver
    book = Resolver()
    align = DiartAttribution(book, None, apply=False)
    for e in events:
        if e["type"] in ("speaker", "revision", "generation"):
            align.on_event(dict(e))
    gen = align.generation
    c = {"words": 0, "other_as_wearer": 0, "wearer_as_other": 0, "right": 0, "unresolved": 0, "in_overlap": 0}
    for text, a, b in words:
        mid = (a + b) / 2
        who, overlap = truth_at(truth, mid)
        if not who:
            continue
        c["words"] += 1
        kind, label = align.word_label(gen, a, b)
        status = align.speakers.get((gen, label), {}).get("wearer") if kind == "single" else None
        if overlap:
            c["in_overlap"] += 1
            if kind == "single" and status == "wearer" and wearer not in who:
                c["other_as_wearer"] += 1
            continue
        if kind != "single" or status not in ("wearer", "other"):
            c["unresolved"] += 1
        elif status == "wearer":
            c["right" if who == [wearer] else "other_as_wearer"] += 1
        else:
            c["right" if who != [wearer] else "wearer_as_other"] += 1
    return c


# ---------------------------------------------------------------- calibration

def calibrate(models, settings, profile, wearer_clips, other_clips):
    used = set(json.load(open(CALIBRATION)).get("files", [])) if os.path.exists(CALIBRATION) else set()
    sims = {"wearer": [], "other": []}
    for kind, paths in (("wearer", wearer_clips), ("other", other_clips)):
        for path in paths:
            events, summary, _ = run(models, settings, read_wav(path), profile)
            for label, s in summary["speakers"].items():
                if s.get("similarity") is not None and s.get("clean_seconds", 0) >= settings.wearer_min_seconds:
                    sims[kind].append(s["similarity"])
            used.add(digest(path))
    if not sims["wearer"] or not sims["other"]:
        sys.exit("Calibration needs session speakers from both wearer and other recordings.")
    lo_w, hi_o = min(sims["wearer"]), max(sims["other"])
    if lo_w > hi_o:
        yes = round(lo_w - 0.25 * (lo_w - hi_o), 3)
        no = round(hi_o + 0.25 * (lo_w - hi_o), 3)
    else:                                                 # overlapping: keep a wide "uncertain" band
        yes, no = round(lo_w + 0.75 * (hi_o - lo_w), 3), round(hi_o - 0.75 * (hi_o - lo_w), 3)
        yes, no = max(yes, hi_o), min(no, lo_w)
    out = {"wearer_similarities": sims["wearer"], "other_similarities": sims["other"],
           "suggested": {"SPEAKER_WEARER_YES": yes, "SPEAKER_WEARER_NO": no}, "files": sorted(used)}
    os.makedirs(os.path.dirname(CALIBRATION), exist_ok=True)
    with open(CALIBRATION, "w") as f:
        json.dump(out, f, indent=1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recording", nargs="?")
    p.add_argument("--truth")
    p.add_argument("--wearer", default="A")
    p.add_argument("--deepgram", help="the recording's Deepgram live messages (.jsonl)")
    p.add_argument("--profile", help="wearer profile (default: SPEAKER_PROFILE)")
    p.add_argument("--stream", action="store_true")
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--latency", type=float)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--wearer-clips", nargs="*", default=[])
    p.add_argument("--other-clips", nargs="*", default=[])
    p.add_argument("--out")
    p.add_argument("--standin-models", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    settings = Settings()
    if args.latency:
        settings.latency = args.latency
    models, startup = load_models(settings, args.standin_models)
    profile = enrollment.load(args.profile or settings.profile, None if args.standin_models else settings)
    print("Models ready in %.1f s (startup, measured separately). Window %.1f s, step %.2f s, latency %.2f s; "
          "wearer profile: %s." % (startup, models.duration, settings.step, models.config.latency,
                                  "yes" if profile else "none"))
    if args.calibrate:
        if profile is None:
            sys.exit("Calibration needs an enrolled wearer profile.")
        out = calibrate(models, settings, profile, args.wearer_clips, args.other_clips)
        print(json.dumps(out["suggested"]), "-> put these in .env. These files are now marked as calibration "
              "data and should not be used for evaluation.")
        return
    if not args.recording:
        p.error("give a recording, or --calibrate")
    if os.path.exists(CALIBRATION) and digest(args.recording) in json.load(open(CALIBRATION)).get("files", []):
        print("WARNING: this recording was used for calibration; its scores are not an evaluation.")
    events, summary, took = run(models, settings, read_wav(args.recording), profile, args.stream, args.realtime)
    result = {"recording": os.path.basename(args.recording), "startup_seconds": startup, "run_seconds": took,
              "summary": summary, "mode": "stream" + ("+realtime" if args.realtime else "") if args.stream else "offline"}
    if args.truth:
        truth = json.load(open(args.truth))
        result["moments"] = score_moments(events, truth, args.wearer)
        if args.deepgram:
            result["words"] = score_words(events, truth, args.wearer, deepgram_words(args.deepgram))
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}, indent=1))
    print("steady state: step %.3f s median / %.3f s p95 per %.2f s of audio; processing lag %.2f s median / %.2f s p95; "
          "algorithmic latency %.2f s" % (summary["step_seconds_median"] or 0, summary["step_seconds_p95"] or 0,
                                          settings.step, summary["processing_lag_median"] or 0,
                                          summary["processing_lag_p95"] or 0, summary["algorithmic_latency"]))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(result, events=events), f)


if __name__ == "__main__":
    main()
