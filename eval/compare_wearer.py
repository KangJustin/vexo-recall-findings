"""Who is the wearer? The existing baseline vs the diart speaker service, on the same labelled conversations.

    python3 eval/compare_wearer.py            (system python; runs the service side in .venv-speaker)

Each scenario's truth names the wearer ("A" in the synthetic set). Words come from the recorded Deepgram live
messages; a word's true speaker is whoever spoke at its middle. Reported per system, at word level:
  other_as_wearer   someone else's word put on the wearer      (the main failure being fixed)
  wearer_as_other   the wearer's word put on someone else
  unresolved        left without a decision (honest, but not right)
  in_overlap        words inside real overlap (other_as_wearer still counts there)
The baseline is the app as it is: Deepgram labels + voiceprints + the enrolled ECAPA wearer baseline. The new
system is speaker_service (diart + pyannote/segmentation-3.0 + ECAPA). Synthetic voices are cleaner than real
people: real recordings (eval/README.md) are the evidence that matters.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
VENV = os.path.join(REPO, ".venv-speaker", "bin", "python")
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
import corpus  # noqa: E402
import voice_baseline  # noqa: E402

WEARER = "A"
RESULTS = os.path.join(HERE, "results")


def truth_of(truth, t):
    who = sorted({x["speaker"] for x in truth["turns"] if x["start"] <= t < x["end"]})
    return who, len(who) > 1 or any(a <= t < b for a, b in truth["overlaps"])


def baseline_profile():
    """The synthetic wearer's enrolled ECAPA baseline (familiar/enroll.py), from the three prompts."""
    path = os.path.join(RESULTS, "baseline_profile_synthetic_wearer.json")
    if not os.path.exists(path):
        import numpy as np
        from familiar import enroll
        from familiar.voices import load_encoder
        enc = load_encoder()
        samples = [enroll.check_sample(voice_baseline.say("Samantha", p))[0] for p in enroll.PROMPTS]
        os.makedirs(RESULTS, exist_ok=True)
        json.dump(enroll.build_profile(samples, enc), open(path, "w"))
    return path


def score_baseline(name, truth):
    out = json.loads(subprocess.run([sys.executable, "-W", "ignore", os.path.join(HERE, "replay.py"), ".", name,
                                     "--me-profile", baseline_profile()], capture_output=True, text=True,
                                    cwd=REPO).stdout.strip().splitlines()[-1])
    me = out["wearer_person"]
    c = {"words": 0, "right": 0, "other_as_wearer": 0, "wearer_as_other": 0, "unresolved": 0, "in_overlap": 0}
    for w in out["words"]:
        who, overlap = truth_of(truth, (w["start"] + w["end"]) / 2)
        if not who:
            continue
        c["words"] += 1
        on_wearer = me is not None and w["speaker"] == me
        if overlap:
            c["in_overlap"] += 1
            if on_wearer and WEARER not in who:
                c["other_as_wearer"] += 1
            continue
        if w["speaker"] == "?":
            c["unresolved"] += 1
        elif on_wearer:
            c["right" if who == [WEARER] else "other_as_wearer"] += 1
        else:
            c["right" if who != [WEARER] else "wearer_as_other"] += 1
    return c


def score_service(name):
    profile = os.path.join(RESULTS, "diart_profile_synthetic_wearer.json")
    cmd = [VENV, "-W", "ignore", "-m", "speaker_service.replay", os.path.join(HERE, "corpus", name + ".wav"),
           "--truth", os.path.join(HERE, "corpus", name + ".json"), "--wearer", WEARER,
           "--deepgram", corpus.messages_path(name), "--profile", profile, "--stream"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    if r.returncode or "{" not in r.stdout:
        reason = (r.stdout + r.stderr).strip().splitlines()
        return None, next((l for l in reversed(reason) if "Models not ready" in l or "HF_TOKEN" in l), reason[-1] if reason else "failed")
    body = r.stdout[r.stdout.index("{"):r.stdout.rindex("}") + 1]
    return json.loads(body).get("words"), None


def main():
    rows = {"baseline": {}, "speaker_service": {}}
    unavailable = None
    for name in corpus.names():
        _, truth = corpus.load(name)
        rows["baseline"][name] = score_baseline(name, truth)
        if unavailable is None:
            words, problem = score_service(name)
            if problem:
                unavailable = problem
            else:
                rows["speaker_service"][name] = words
        print("done", name, flush=True)
    lines = ["| system | words | right | other-as-wearer | wearer-as-other | unresolved | in overlap |", "|---|---|---|---|---|---|---|"]
    for system, per in rows.items():
        if not per:
            lines.append("| %s | not run: %s | | | | | |" % (system, unavailable))
            continue
        t = {k: sum(c[k] for c in per.values()) for k in ("words", "right", "other_as_wearer", "wearer_as_other",
                                                         "unresolved", "in_overlap")}
        lines.append("| %s | %d | %d | %d | %d | %d | %d |" % (system, t["words"], t["right"], t["other_as_wearer"],
                                                             t["wearer_as_other"], t["unresolved"], t["in_overlap"]))
    lines += ["", "Per scenario (other-as-wearer / wearer-as-other / unresolved):"]
    for system, per in rows.items():
        if per:
            lines.append("- %s: %s" % (system, ", ".join("%s %d/%d/%d" % (n, c["other_as_wearer"], c["wearer_as_other"],
                                                                         c["unresolved"]) for n, c in per.items())))
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(RESULTS, "compare_wearer.md"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
