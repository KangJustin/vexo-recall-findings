# Speaker service (experimental, behind `--speaker-service`)

Streaming diarization with **diart 0.9.2**, segmentation **pyannote/segmentation-3.0** and embeddings
**speechbrain/spkrec-ecapa-voxceleb**, plus wearer matching against an enrolled profile. It runs as its own
FastAPI process in its own Python environment; the app streams the band's audio to it and keeps transcribing
with Deepgram whatever the service does.

## Where it sits

```
band --Opus 20 ms--> run.py --same packets--> Deepgram live (words + timestamps)
                        |
                        +-- decoded 16 kHz mono PCM (no made-up audio for lost packets, so this clock = Deepgram's)
                              +--> voice matching (familiar/voices.py, the existing baseline)
                              +--> familiar/speaker_client.py --WebSocket--> speaker_service (diart)
                                                                                  |
              familiar/diart_align.py <--speaker events (session time)------------+
              aligns events with Deepgram's words -> reversible corrections on the page
```

Session speaker ids (`s0`, `s1`, ...) belong to one conversation and one diarization *generation*; they are never
people. Only the enrolled wearer profile is a saved identity. A session speaker is `wearer` / `other` /
`uncertain` / `pending` from its clustering center (built by diart from overlap-penalised embeddings) vs the
profile. Overlap is taken from the raw segmentation, so it is flagged even when diart cannot identify one of the
voices; words in overlap stay unresolved with candidates.

## Setup (once)

```
cd ~/vexo-recall
python3.11 -m venv .venv-speaker
.venv-speaker/bin/python -m pip install -r speaker_service/requirements.txt
```

Model access: `pyannote/segmentation-3.0` is gated.
1. Sign in at huggingface.co, open https://huggingface.co/pyannote/segmentation-3.0 and accept the conditions.
2. Settings -> Access Tokens -> create a **read** token.
3. Add `HF_TOKEN=hf_...` to `~/vexo-recall/.env` (see `.env.example`). Never in code, chats or logs.
The ECAPA model is public. Weights are cached by Hugging Face / SpeechBrain after the first download.

## Start

```
.venv-speaker/bin/uvicorn speaker_service.app:app --host 127.0.0.1 --port 8790
curl http://127.0.0.1:8790/ready          # 200 when ready; 503 with the exact problem otherwise
```

Enrol the wearer (alone, quiet, band worn as usual; each clip is checked with the service's own models:
silence, clipping, noise, a second voice):
```
python3 run.py 64 --enroll-service http://127.0.0.1:8790
```

Live, with the new attribution path (off unless this flag is given):
```
python3 run.py 64 --speaker-service ws://127.0.0.1:8790
```
The page shows "Speaker AI on" or "Speaker AI unavailable" (service down, not ready, or more than
SPEAKER_MAX_LAG seconds behind: then the backlog is skipped, reported as a gap, and a new generation starts).

## Interface

`GET /health`, `GET /ready`, `GET|DELETE /enrollment`, `POST /enrollment/clips` (WAV body), `POST /enrollment/commit`,
`GET /sessions/{id}`, `WS /sessions/{id}/audio`. Events: `speaker` (session_id, generation, revision, start, end,
segments with speakers / overlap / unidentified / wearer, processing_lag, algorithmic_latency, attribution),
`revision` (a session speaker's wearer status changed; applies_from), `status`, `gap`, `generation`, `summary`.
Protocol details: the docstring of `speaker_service/app.py`.

## Replay and evaluation

```
# 1. record a real conversation, with everyone's consent (saves .wav + Deepgram .jsonl)
python3 run.py 64 --record eval/corpus/talk1            # then write eval/corpus/talk1.json (eval/README.md)
# 2. whole recording, then as live 20 ms chunks at real-time pace
.venv-speaker/bin/python -m speaker_service.replay eval/corpus/talk1.wav --truth eval/corpus/talk1.json --wearer A \
    --deepgram eval/corpus/talk1.jsonl
.venv-speaker/bin/python -m speaker_service.replay eval/corpus/talk1.wav --truth eval/corpus/talk1.json --wearer A \
    --deepgram eval/corpus/talk1.jsonl --stream --realtime
# 3. calibrate thresholds on SEPARATE recordings (they are then flagged if reused for evaluation)
.venv-speaker/bin/python -m speaker_service.replay --calibrate --wearer-clips me1.wav me2.wav --other-clips them1.wav them2.wav
# 4. baseline vs speaker service, word level
python3 eval/compare_wearer.py
```

## Tests

```
.venv-speaker/bin/python -m unittest discover -s speaker_service/tests -t .     # service mechanics
python3 -m unittest discover -s tests                                         # app side, incl. end-to-end
```
These use STAND-IN models (tones) with diart's real clustering and aggregation: they check timing, events,
state, overlap flags, overload, reconnects, enrolment checks and the protocol. They say nothing about accuracy
on real voices.
