"""Replay recorded Deepgram LIVE messages + the scenario audio through the app's own pipeline, offline.

    python3 eval/replay.py <code_root> <scenario> [--no-voices] [--second-pass ENGINE --apply]

Prints JSON: the words as the APP finally attributed them ({"text", "start", "end", "speaker"}, speaker "?"
for unresolved), plus timing. <code_root> is a folder holding a `familiar/` package, so an older copy of the
code can be measured on exactly the same input. The audio is fed in step with the messages (each message
says how much audio had been sent when it arrived), as in the live app.
"""
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    root, name = sys.argv[1], sys.argv[2]
    no_voices = "--no-voices" in sys.argv
    engine = sys.argv[sys.argv.index("--second-pass") + 1] if "--second-pass" in sys.argv else None
    sys.path.insert(0, os.path.abspath(root))
    sys.path.insert(1, HERE)
    sys.path.insert(2, os.path.join(REPO, "vexo-open-kit", "examples", "python"))
    import corpus
    import numpy as np
    from familiar import voices as V
    from familiar.live import DiarizedDeepgram
    from familiar.resolver import Resolver

    audio, truth = corpus.load(name)
    msgs = [json.loads(l) for l in open(corpus.messages_path(name))]
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    book = Resolver()
    new_api = "connection" in V.VoicePrints.settle.__code__.co_varnames
    me = None
    if "--me-profile" in sys.argv:                 # an enrolled wearer baseline (familiar/enroll.py format)
        me = json.load(open(sys.argv[sys.argv.index("--me-profile") + 1]))
        me["vector"] = np.asarray(me["embedding"], dtype=np.float32)
    if no_voices and new_api:
        vp = V.VoicePrints(book, enabled=False)
    else:
        kw = {"me": me} if me is not None and new_api else {}
        vp = V.VoicePrints(book, encoder=V.load_encoder(), log=lambda _: None, **kw)
        vp.ready.set()
    second = None
    review = "--review" in sys.argv
    if (engine or review) and new_api:
        from familiar import second_pass
        second = second_pass.SecondPass(book, vp, engine=None if engine == "none" else engine,
                                        apply="--apply" in sys.argv, threaded=False,
                                        review=second_pass.GeminiReview(use_files="--files" in sys.argv) if review else None)
        vp.on_segment = second.note
    if new_api:
        ears = DiarizedDeepgram(on_sentence=vp.settle, on_interim=vp.interim)
    else:
        ears = DiarizedDeepgram(on_piece=lambda s, t, f: book.feed(vp.interim_voice(s), t, f) if not f else None,
                                on_sentence=vp.settle)
    ears.finals = []
    fed = 0
    work = 0.0
    for m in msgs:
        upto = int(m["audio_sent"] * corpus.SR) * 2
        if upto > fed:
            vp.add_audio(pcm[fed:upto])
            fed = upto
        ears._message(m["message"])
        while not vp._jobs.empty():
            t0 = time.perf_counter()
            vp.process(*vp._jobs.get())
            work += time.perf_counter() - t0
    vp.add_audio(pcm[fed:])
    ears._sentence_ends()
    while not vp._jobs.empty():
        t0 = time.perf_counter()
        vp.process(*vp._jobs.get())
        work += time.perf_counter() - t0
    if second:
        t0 = time.perf_counter()
        second.flush(final=True)
        second_time = time.perf_counter() - t0
    log = book.snapshot()["log"]
    if new_api:
        words = [{"text": w[0], "start": w[1], "end": w[2], "speaker": "?" if line["speaker"] is None else line["speaker"]}
                 for line in log for w in line.get("words") or []]
    else:                                    # old code kept no word timings: align line tokens with the final words
        finals = [w for m in msgs if m["message"].get("type") == "Results" and m["message"].get("is_final")
                  for w in m["message"]["channel"]["alternatives"][0].get("words", [])]
        tokens = [(line["speaker"], tok) for line in log for tok in line["text"].split()]
        words = [{"text": f.get("punctuated_word") or f["word"], "start": f["start"], "end": f["end"], "speaker": sp}
                 for (sp, _), f in zip(tokens, finals)]
    out = {"words": words, "voice_seconds": round(work, 3), "lines": len(log),
           "people": len(book.snapshot()["people"]),
           "wearer_person": next((p["id"] for p in book.snapshot()["people"] if p["is_me"]), None)}
    if second:
        out.update(second.report(), second_seconds=round(second_time, 3))
        if second.review is not None:
            st = second.review.stats
            lat = sorted(st["latency"])
            out["review"] = {"requests": st["requests"], "uploads": st["uploads"], "reused_uploads": st["reused_uploads"],
                             "input_tokens": st["input_tokens"], "output_tokens": st["output_tokens"],
                             "latency_median": lat[len(lat) // 2] if lat else None,
                             "proposals": [{k: p[k] for k in ("line", "words", "to", "start", "end")} for r in second.reviews
                                           for p in r["proposals"]],
                             "unresolved_regions": sum(len(r["unresolved"]) for r in second.reviews),
                             "overlap_segments": sum(1 for r in second.reviews for g in r["segments"]
                                                     if g.get("overlap_status") in ("overlap", "possible")),
                             "dropped": [d for r in second.reviews for d in r["dropped"]]}
            # the words if every review proposal were applied (hypothetical, for scoring only)
            if second.review is not None:
                ids = [p["id"] for p in book.proposals if p["engine"] == "gemini_review"]
                book.apply_proposals(ids)
                out["words_if_review_applied"] = [
                    {"text": w[0], "start": w[1], "end": w[2], "speaker": "?" if line["speaker"] is None else line["speaker"]}
                    for line in book.snapshot()["log"] for w in line.get("words") or []]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
