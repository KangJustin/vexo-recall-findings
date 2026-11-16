"""Score word-level speaker attribution against a scenario's ground truth.

A system's output is a list of words: {"text", "start", "end", "speaker"} where speaker is the system's own
label (any hashable), or None / "?" for "unresolved". Labels are matched to the true speakers one-to-one in
the way that gives the most correct words (so a person split into two labels still costs errors).

Per scenario it reports:
  correct     words given the right speaker
  wrong       words given a wrong speaker (includes a split person's second label)
  unresolved  words the system left without a speaker (honest uncertainty, not counted as wrong)
  overlap     words inside a region where two people really spoke at once: either speaker counts as right
  missed      true words with no transcribed word in their turn (overlapping speech often loses words)
  short       the brief interjections ("wait", "yeah"): transcribed? attributed correctly?
"""
import itertools
import re

TOL = 0.08          # seconds of timestamp jitter allowed at turn edges


def _norm(word):
    return re.sub(r"[^a-z']", "", word.lower())


def truth_speakers(truth, mid):
    """Set of true speakers at time `mid` (two inside an overlap), nearest turn if in a gap."""
    inside = {t["speaker"] for t in truth["turns"] if t["start"] - TOL <= mid <= t["end"] + TOL}
    if inside:
        return inside
    nearest = min(truth["turns"], key=lambda t: min(abs(mid - t["start"]), abs(mid - t["end"])))
    return {nearest["speaker"]}


def in_overlap(truth, mid):
    return any(a - TOL <= mid <= b + TOL for a, b in truth["overlaps"])


def best_mapping(words, truth):
    labels = sorted({w["speaker"] for w in words if w["speaker"] not in (None, "?")}, key=str)
    people = sorted({t["speaker"] for t in truth["turns"]})
    best, best_score = {}, -1
    for k in range(min(len(labels), len(people)) + 1):
        for chosen in itertools.permutations(people, k):
            for subset in itertools.combinations(labels, k):
                mapping = dict(zip(subset, chosen))
                score = sum(1 for w in words if mapping.get(w["speaker"]) in
                            truth_speakers(truth, (w["start"] + w["end"]) / 2))
                if score > best_score:
                    best, best_score = mapping, score
    return best


def score(words, truth):
    mapping = best_mapping(words, truth)
    out = {"words": len(words), "correct": 0, "wrong": 0, "unresolved": 0, "overlap": 0,
           "labels": len({w["speaker"] for w in words if w["speaker"] not in (None, "?")}),
           "true_speakers": len({t["speaker"] for t in truth["turns"]}), "errors": []}
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        true = truth_speakers(truth, mid)
        if in_overlap(truth, mid):
            out["overlap"] += 1
        if w["speaker"] in (None, "?"):
            out["unresolved"] += 1
        elif mapping.get(w["speaker"]) in true:
            out["correct"] += 1
        else:
            out["wrong"] += 1
            out["errors"].append("%.2f %s: said by %s, labelled %s" % (
                w["start"], w["text"], "/".join(sorted(true)), mapping.get(w["speaker"], "extra label %s" % w["speaker"])))
    # missed words and the short interjections, turn by turn
    out["missed"], out["short"] = 0, []
    for turn in truth["turns"]:
        tokens = [t for t in (_norm(x) for x in turn["text"].split()) if t]
        heard = [w for w in words if turn["start"] - TOL <= (w["start"] + w["end"]) / 2 <= turn["end"] + TOL]
        found = [w for w in heard if _norm(w["text"]) in tokens]
        out["missed"] += max(0, len(tokens) - len({_norm(w["text"]) for w in found}))
        if len(tokens) <= 2:
            hit = [w for w in found]
            out["short"].append({
                "text": turn["text"], "speaker": turn["speaker"], "transcribed": bool(hit),
                "attributed": ("unresolved" if hit and all(w["speaker"] in (None, "?") for w in hit) else
                               bool(hit) and any(mapping.get(w["speaker"]) == turn["speaker"] for w in hit))})
    return out


def summary_line(name, s):
    short = ", ".join("%s:%s" % (x["text"], "missed" if not x["transcribed"] else
                                 ("ok" if x["attributed"] is True else
                                  ("unresolved" if x["attributed"] == "unresolved" else "WRONG"))) for x in s["short"])
    return "%-18s words %3d  correct %3d  wrong %3d  unresolved %3d  missed %3d  labels %d/%d  %s" % (
        name, s["words"], s["correct"], s["wrong"], s["unresolved"], s["missed"], s["labels"], s["true_speakers"],
        ("short[" + short + "]") if short else "")
