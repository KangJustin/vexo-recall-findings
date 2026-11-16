"""Run every attribution system on the labelled scenarios and write eval/results/report.md.

    python3 eval/corpus.py           # once: make the scenarios
    python3 eval/deepgram_live.py    # once: record Deepgram LIVE output for them (real API, ~20 s)
    python3 eval/deepgram_batch.py   # once: Deepgram recorded-audio diarization on whole clips (reference)
    python3 eval/run.py [--before PATH] [--skip-gemini]

Systems compared on identical input:
  deepgram_live      Deepgram's own live word labels, unprocessed
  app_before         the app before this change (copy of the old familiar/ given with --before)
  app_no_voices      the app with voice matching off: Deepgram labels mapped to people
  app                the app as shipped: labels first, voiceprints as a careful check
  app+2nd_suggest    + Deepgram recorded-audio second pass, suggestions only (= what is shown)
  app+2nd_apply      + second pass applying what Deepgram is sure of
  app+gemini_review  + Gemini audio review, IF every validated suggestion were applied (hypothetical)
  deepgram_batch     Deepgram recorded-audio diarization on the WHOLE clip (a reference, not a live option)
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
import corpus  # noqa: E402
import score  # noqa: E402
from familiar import second_pass  # noqa: E402


def replay(root, name, *flags):
    r = subprocess.run([sys.executable, "-W", "ignore", os.path.join(HERE, "replay.py"), root, name] + list(flags),
                       capture_output=True, text=True, cwd=REPO)
    if r.returncode:
        raise RuntimeError("%s %s %s: %s" % (root, name, flags, r.stderr[-500:]))
    return json.loads(r.stdout.strip().splitlines()[-1])


def live_words(name):
    words, latency = [], []
    for line in open(corpus.messages_path(name)):
        m = json.loads(line)
        r = m["message"]
        if r.get("type") == "Results" and r.get("is_final"):
            for w in r["channel"]["alternatives"][0].get("words", []):
                words.append({"text": w.get("punctuated_word") or w["word"], "start": w["start"], "end": w["end"],
                              "speaker": w.get("speaker")})
                latency.append(m["received"] - w["end"])
    return words, latency


def main():
    before = sys.argv[sys.argv.index("--before") + 1] if "--before" in sys.argv else None
    skip_gemini = "--skip-gemini" in sys.argv
    systems = {}
    usage = {}

    def add(system, name, words, truth, people=None):
        s = score.score(words, truth)
        s["people"] = people
        systems.setdefault(system, {})[name] = s

    live_latency = []
    for name in corpus.names():
        _, truth = corpus.load(name)
        words, lat = live_words(name)
        live_latency += lat
        add("deepgram_live", name, words, truth, len({w["speaker"] for w in words}))
        raw = json.load(open(os.path.join(HERE, "results", "deepgram_batch", name + ".json")))
        bw = second_pass.words_from_deepgram_batch(raw["response"])
        add("deepgram_batch", name, bw, truth, len({w["speaker"] for w in bw}))
        usage.setdefault("deepgram_batch", []).append(raw["seconds_taken"])
        if before:
            out = replay(before, name)
            add("app_before", name, out["words"], truth, out["people"])
        out = replay(".", name, "--no-voices")
        add("app_no_voices", name, out["words"], truth, out["people"])
        out = replay(".", name)
        add("app", name, out["words"], truth, out["people"])
        usage.setdefault("app_voice_cpu_s_per_line", []).append(out["voice_seconds"] / max(1, out["lines"]))
        out = replay(".", name, "--second-pass", "deepgram")
        add("app+2nd_suggest", name, out["words"], truth, out["people"])
        usage.setdefault("2nd_pass", []).append(out)
        out = replay(".", name, "--second-pass", "deepgram", "--apply")
        add("app+2nd_apply", name, out["words"], truth, out["people"])
        if not skip_gemini:
            out = replay(".", name, "--second-pass", "none", "--review")
            add("app+gemini_review", name, out["words_if_review_applied"], truth, out["people"])
            usage.setdefault("review", []).append(out["review"])
        print("done", name, flush=True)

    # Gemini 3.5 Transcribe: available to this project?
    try:
        audio, _ = corpus.load("no_silence")
        second_pass.gemini_transcribe(second_pass.wav_bytes(audio))
        transcribe = "available (not yet scored)"
    except Exception as problem:
        transcribe = "not available to this project: %s" % problem

    rows = []
    for system, per in systems.items():
        tot = {k: sum(s[k] for s in per.values()) for k in ("words", "correct", "wrong", "unresolved", "missed")}
        short = [x for s in per.values() for x in s["short"]]
        tot["short_ok"] = sum(1 for x in short if x["attributed"] is True)
        tot["short_wrong"] = sum(1 for x in short if x["transcribed"] and x["attributed"] is False)
        tot["short_unresolved"] = sum(1 for x in short if x["attributed"] == "unresolved")
        tot["short_missed"] = sum(1 for x in short if not x["transcribed"])
        tot["extra_cards"] = sum(max(0, (s["people"] or 0) - s["true_speakers"]) for s in per.values())
        rows.append((system, tot, per))
    lat = sorted(live_latency)
    sp_lat = sorted(l for o in usage["2nd_pass"] for l in ([o["latency_median"]] if o["latency_median"] else []))
    lines = ["# Speaker attribution evaluation", "", "Generated %s by eval/run.py on %d synthetic scenarios "
             "(macOS text-to-speech voices; ground truth from the script). Synthetic voices are cleaner and more distinct "
             "than real people on a wrist mic: this measures how the pipeline handles each CASE, not real-world "
             "accuracy." % (time.strftime("%Y-%m-%d %H:%M"), len(corpus.names())), "",
             "| system | wrong | unresolved | correct | missed words | interjections ok / wrong / unresolved / missed | extra cards |",
             "|---|---|---|---|---|---|---|"]
    for system, t, _ in rows:
        lines.append("| %s | %d | %d | %d / %d | %d | %d / %d / %d / %d | %d |" % (
            system, t["wrong"], t["unresolved"], t["correct"], t["words"], t["missed"], t["short_ok"], t["short_wrong"],
            t["short_unresolved"], t["short_missed"], t["extra_cards"]))
    lines += ["", "## Per scenario (wrong / unresolved)", "", "| system | " + " | ".join(corpus.names()) + " |",
              "|---|" + "---|" * len(corpus.names())]
    for system, _, per in rows:
        lines.append("| %s | %s |" % (system, " | ".join("%d / %d" % (per[n]["wrong"], per[n]["unresolved"])
                                                         for n in corpus.names())))
    review = usage.get("review", [])
    lines += ["", "## Latency and API usage", "",
              "- Deepgram live: final word arrives %.2f s (median) / %.2f s (p90) after it was spoken." % (
                  lat[len(lat) // 2], lat[int(len(lat) * 0.9)]),
              "- App voice matching: %.0f ms CPU per settled line (median scenario)." % (
                  1000 * sorted(usage["app_voice_cpu_s_per_line"])[len(usage["app_voice_cpu_s_per_line"]) // 2]),
              "- Deepgram second pass: %d requests, %.0f s of audio sent, request latency median %s s; a correction "
              "can only come %.0f s after the risky moment (the AFTER context) plus that." % (
                  sum(o["requests"] for o in usage["2nd_pass"]), sum(o["audio_seconds_sent"] for o in usage["2nd_pass"]),
                  sp_lat[len(sp_lat) // 2] if sp_lat else "n/a", second_pass.AFTER),
              "- Deepgram whole-clip reference: %.2f s per 10-20 s clip." % (sorted(usage["deepgram_batch"])[len(usage["deepgram_batch"]) // 2])]
    if review:
        rl = sorted(r["latency_median"] for r in review if r["latency_median"])
        lines.append("- Gemini review (%s): %d requests, %d input + %d output/thinking tokens, latency median %s s, "
                     "%d unresolved regions and %d overlap segments reported, dropped by validation: %s." % (
                         second_pass.GEMINI_REVIEW_MODEL, sum(r["requests"] for r in review),
                         sum(r["input_tokens"] for r in review), sum(r["output_tokens"] for r in review),
                         rl[len(rl) // 2] if rl else "n/a", sum(r["unresolved_regions"] for r in review),
                         sum(r["overlap_segments"] for r in review), sum(len(r["dropped"]) for r in review)))
    lines.append("- Gemini 3.5 Transcribe: %s" % transcribe)
    lines += ["", "## Errors (first few per system)", ""]
    for system, _, per in rows:
        errs = ["%s: %s" % (n, e) for n, s in per.items() for e in s["errors"][:2]]
        if errs:
            lines.append("- **%s**: %s" % (system, "; ".join(errs[:6])))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    report = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "results", "report.md"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
