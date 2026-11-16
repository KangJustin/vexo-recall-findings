# Speaker attribution: evaluation

`eval/` measures who-said-what on labelled audio, with every system on identical input.

```
python3 eval/corpus.py           # synthetic scenarios (macOS voices) with ground truth -> eval/corpus/
python3 eval/deepgram_live.py    # stream each one to Deepgram LIVE like the band does (Opus, real time)
python3 eval/deepgram_batch.py   # Deepgram recorded-audio diarization on the whole clip (reference)
python3 eval/run.py              # replay through the app, score everything -> eval/results/report.md
```

`eval/run.py --skip-gemini` leaves out the Gemini review (it costs tokens). `--before PATH` also scores an
older copy of the code (a folder holding a `familiar/` package).

## Real conversations (what the synthetic set cannot tell you)

Synthetic voices are clean and very distinct; real people on a wrist mic are not. Results on real speech can
differ a lot (14 Nov: same-person voiceprint similarity ~0.4-0.65 live, vs ~0.8 on synthetic voices).

1. **With everyone's consent**, record a conversation:
   `python3 run.py 64 --record eval/corpus/talk1` (saves `talk1.wav` and Deepgram's raw `talk1.jsonl`).
2. Write `eval/corpus/talk1.json` with who spoke when (seconds from the start of the recording):
   ```json
   {"turns": [{"speaker": "A", "start": 0.8, "end": 3.1, "text": "So what do you build?"},
              {"speaker": "B", "start": 3.0, "end": 6.4, "text": "Grippers, mostly."}],
    "overlaps": [[3.0, 3.1]]}
   ```
   Listen in any audio editor and note the times; `overlaps` lists where two people really talk at once.
3. `python3 eval/deepgram_batch.py talk1` then `python3 eval/run.py`: the recording is scored with the rest.
4. Delete the recording when you are done. `eval/corpus/` and `eval/results/` are git-ignored.

## What the columns mean

- **wrong**: a word given to the wrong person (a person split over two cards counts as wrong).
- **unresolved**: a word the app left as "Unclear who". Honest, but it is a cost: it is not counted as right.
- **missed words**: said but never transcribed (overlapping speech often loses words, in every engine tried).
- **interjections**: the short "wait" / "yeah" / "right" turns: attributed right / wrong / unresolved / missed.
- **extra cards**: more people on the page than really spoke.
