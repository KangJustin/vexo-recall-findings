"""Score the real-time voice tracker (familiar/tracker.py) on the labelled synthetic scenarios.

    python3 eval/tracker_eval.py      -> prints the scores, writes eval/results/tracker_synthetic.json

"Samantha" (speaker A, the wearer) is enrolled with the three prompts; each scenario is then streamed through the
tracker in HOP-sized steps, as live. What the tracker SHOWS at each moment is compared with who is really talking
at that moment (so the reaction delay counts against it). Overlap regions and silences are left out.
"""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import corpus  # noqa: E402
import voice_baseline  # noqa: E402
from familiar import enroll, tracker  # noqa: E402
from familiar.voices import load_encoder  # noqa: E402


def truth_at(truth, t):
    who = [x["speaker"] for x in truth["turns"] if x["start"] <= t <= x["end"]]
    if len(who) != 1 or any(a <= t <= b for a, b in truth["overlaps"]):
        return None                                    # silence or overlap: not scored
    return who[0]


def run_scenario(name, encode, profile, **settings):
    audio, truth = corpus.load(name)
    clock = [0.0]
    tr = tracker.VoiceTracker(encode, me=profile, clock=lambda: clock[0], **settings)
    hop = int(tr.hop * corpus.SR)
    shown = []
    for i in range(0, len(audio) - hop + 1, hop):
        tr.feed(audio[i:i + hop])
        clock[0] = (i + hop) / corpus.SR
        shown.append((clock[0], tr.step()["speaker"]))
    return shown, truth


def score(shown, truth):
    """Map session voices (p1, p2...) to true speakers B, C one-to-one in the best way, then count."""
    others = sorted({s for _, s in shown if s.startswith("p")})
    people = sorted({x["speaker"] for x in truth["turns"]} - {"A"})
    best = None
    for k in range(min(len(others), len(people)) + 1):
        for chosen in itertools.permutations(people, k):
            for subset in itertools.combinations(others, k):
                mapping = dict(zip(subset, chosen), you="A")
                ok = sum(1 for t, s in shown if truth_at(truth, t) and mapping.get(s) == truth_at(truth, t))
                if best is None or ok > best[0]:
                    best = (ok, mapping)
    mapping = best[1]
    counts = {"scored": 0, "right": 0, "wrong": 0, "unclear": 0, "silence_shown": 0}
    for t, s in shown:
        real = truth_at(truth, t)
        if real is None:
            continue
        counts["scored"] += 1
        if s in ("unclear", "unknown"):
            counts["unclear"] += 1
        elif s == "silence":
            counts["silence_shown"] += 1
        elif mapping.get(s) == real:
            counts["right"] += 1
        else:
            counts["wrong"] += 1
    # reaction delay: from the start of each turn (of 1.5 s or more) to the first moment it is shown right
    delays = []
    for turn in truth["turns"]:
        if turn["end"] - turn["start"] < 1.5:
            continue
        hit = next((t for t, s in shown if turn["start"] <= t <= turn["end"] + 1.0 and mapping.get(s) == turn["speaker"]), None)
        delays.append(None if hit is None else round(hit - turn["start"], 2))
    counts["delays"] = delays
    counts["people_created"] = len(others)
    counts["true_others"] = len(people)
    return counts


def main():
    encode = load_encoder()
    samples = [enroll.check_sample(voice_baseline.say("Samantha", p))[0] for p in enroll.PROMPTS]
    profile = enroll.build_profile(samples, encode)
    profile["vector"] = np.asarray(profile["embedding"], np.float32)
    print("baseline threshold %.2f (tracker uses %.2f for 1.5 s windows)" % (profile["threshold"],
                                                                        profile["threshold"] - tracker.ME_MARGIN))
    results, total = {}, {"scored": 0, "right": 0, "wrong": 0, "unclear": 0, "silence_shown": 0}
    all_delays = []
    for name in corpus.names():
        shown, truth = run_scenario(name, encode, profile)
        r = score(shown, truth)
        results[name] = r
        for k in total:
            total[k] += r[k]
        all_delays += [d for d in r["delays"] if d is not None]
        print("%-18s right %3d  wrong %2d  unclear %2d  silence-shown %2d  of %3d | people made %d (true %d) | "
              "turn delays %s" % (name, r["right"], r["wrong"], r["unclear"], r["silence_shown"], r["scored"],
                                  r["people_created"], r["true_others"], r["delays"]))
    d = sorted(all_delays)
    print("TOTAL right %.0f%%, wrong %.0f%%, unclear %.0f%%, silence shown %.0f%% of %d moments | delay median %.1f s, "
          "max %.1f s, turns never shown right: %d" % (
              100 * total["right"] / total["scored"], 100 * total["wrong"] / total["scored"],
              100 * total["unclear"] / total["scored"], 100 * total["silence_shown"] / total["scored"], total["scored"],
              d[len(d) // 2], d[-1], sum(1 for r in results.values() for x in r["delays"] if x is None)))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "tracker_synthetic.json"), "w") as f:
        json.dump({"total": total, "scenarios": results}, f, indent=1)


def grid(encode, profile):
    """Try window / hop / smoothing / confirmation settings on all scenarios."""
    rows = []
    for window in (1.0, 1.5):
        for hop in (0.25, 0.5):
            for smooth in (1, 2, 3):
                for confirm in (1, 2):
                    tot = {"scored": 0, "right": 0, "wrong": 0, "unclear": 0}
                    delays, never = [], 0
                    for name in corpus.names():
                        shown, truth = run_scenario(name, encode, profile, window=window, hop=hop, smooth=smooth,
                                                    confirm=confirm)
                        r = score(shown, truth)
                        for k in tot:
                            tot[k] += r[k]
                        delays += [d for d in r["delays"] if d is not None]
                        never += sum(1 for d in r["delays"] if d is None)
                    d = sorted(delays)
                    rows.append((tot["right"] / tot["scored"], window, hop, smooth, confirm, tot["wrong"] / tot["scored"],
                                 tot["unclear"] / tot["scored"], d[len(d) // 2] if d else None, never))
    rows.sort(reverse=True)
    print("right  wrong unclear  delay  never | window hop smooth confirm")
    for r in rows:
        print("%4.0f%%  %4.0f%%  %4.0f%%  %5s  %3d   | %.1f    %.2f  %d      %d" % (
            100 * r[0], 100 * r[5], 100 * r[6], r[7], r[8], r[1], r[2], r[3], r[4]))


if __name__ == "__main__" and "--grid" in sys.argv:
    _enc = load_encoder()
    _samples = [enroll.check_sample(voice_baseline.say("Samantha", p))[0] for p in enroll.PROMPTS]
    _profile = enroll.build_profile(_samples, _enc)
    _profile["vector"] = np.asarray(_profile["embedding"], np.float32)
    grid(_enc, _profile)
elif __name__ == "__main__":
    main()
