"""Streaming speaker diarization (diart + pyannote/segmentation-3.0 + SpeechBrain ECAPA) and wearer matching.

Models load ONCE (Models) and are shared; each conversation has its own Session with its own diart pipeline
(clustering centers, buffers), so speaker state never leaks between conversations.

Timeline: a session's clock is the band's audio clock, in seconds since the session's first sample, the same
clock Deepgram's word timestamps use (the app decodes the same Opus packets it sends to Deepgram).

Speaker ids in events ("s0", "s1", ...) are SESSION ids from diart's clustering; they are not people. Whether a
session speaker is the enrolled wearer is decided only from its clustering center (built by diart from
overlap-penalised embeddings, i.e. mostly clean single-speaker frames) and the enrolled profile:
  similarity >= wearer_yes -> "wearer";  <= wearer_no -> "other";  in between -> "uncertain";
  too little clean speech yet -> "pending";  no enrolled profile -> "unknown".
"""
import itertools
import threading
import time

import numpy as np

_GENERATIONS = itertools.count(1)
INFERENCE_LOCK = threading.Lock()            # models are shared: one forward pass at a time


class ModelAccessError(Exception):
    """A model could not be loaded; the message says what to do (never contains the token)."""


def _access_message(name, problem, has_token):
    text = str(problem)
    if not has_token:
        return ("%s is a gated model and HF_TOKEN is not set. Accept its conditions at https://huggingface.co/%s, "
                "create a read token, and put HF_TOKEN=... in the repository's .env." % (name, name))
    if any(code in text for code in ("401", "403", "Gated", "gated", "restricted", "Unauthorized")):
        return ("HF_TOKEN cannot access %s (%s). Accept the conditions at https://huggingface.co/%s with the account "
                "that owns the token." % (name, type(problem).__name__, name))
    return "Could not load %s: %s" % (name, type(problem).__name__)


class Models:
    """Loads the two requested models explicitly (no diart default can slip in) and checks them."""

    def __init__(self, settings, segmentation_loader=None, embedding_loader=None):
        self.settings = settings
        self._segmentation_loader = segmentation_loader
        self._embedding_loader = embedding_loader
        self.status, self.error, self.startup_seconds = "not loaded", None, None
        self.duration = None
        self.specs = {}
        self.config = None

    def load(self):
        started = time.monotonic()
        self.status = "loading"
        try:
            import torch
            from diart import SpeakerDiarizationConfig
            from diart.models import EmbeddingModel, SegmentationModel
            device = torch.device(self.settings.device)
            seg_module = (self._segmentation_loader or self._load_segmentation)()
            emb_module = (self._embedding_loader or self._load_embedding)()
            segmentation = SegmentationModel(lambda: seg_module)
            embedding = EmbeddingModel(lambda: emb_module)
            segmentation.load()
            embedding.load()
            self.duration = float(self.specs.get("duration") or 10.0)
            latency = min(max(self.settings.latency, self.settings.step), self.duration)
            self.config = SpeakerDiarizationConfig(
                segmentation=segmentation, embedding=embedding, duration=self.duration, step=self.settings.step,
                latency=latency, tau_active=self.settings.tau_active, rho_update=self.settings.rho_update,
                delta_new=self.settings.delta_new, gamma=self.settings.gamma, beta=self.settings.beta,
                max_speakers=self.settings.max_speakers, device=device, sample_rate=self.settings.sample_rate)
            self._warm_up()
            self.status = "ready"
        except ModelAccessError as problem:
            self.status, self.error = "error", str(problem)
        except Exception as problem:                     # report clearly, never the token
            text = (str(problem).splitlines() or [""])[0][:200]
            if self.settings.hf_token:
                text = text.replace(self.settings.hf_token, "***")
            self.status, self.error = "error", "Model loading failed: %s: %s" % (type(problem).__name__, text)
        self.startup_seconds = round(time.monotonic() - started, 2)
        return self

    def _load_segmentation(self):
        name, token = self.settings.segmentation, self.settings.hf_token
        try:
            from pyannote.audio import Model
            from diart.models import PowersetAdapter
            model = Model.from_pretrained(name, use_auth_token=token or False)
        except Exception as problem:
            raise ModelAccessError(_access_message(name, problem, bool(token))) from None
        if model is None:                                # pyannote returns None for some access problems
            raise ModelAccessError(_access_message(name, "403 gated", bool(token)))
        specs = model.specifications
        self.specs = {"duration": float(specs.duration), "powerset": bool(specs.powerset),
                      "classes": len(specs.classes), "powerset_max_classes": getattr(specs, "powerset_max_classes", None),
                      "sample_rate": int(model.audio.sample_rate), "mono": model.audio.mono}
        if self.specs["sample_rate"] != self.settings.sample_rate:
            raise ModelAccessError("%s expects %d Hz audio, the service produces %d Hz" % (
                name, self.specs["sample_rate"], self.settings.sample_rate))
        # segmentation-3.0 outputs powerset classes; diart's adapter turns them into per-speaker activations
        return PowersetAdapter(model) if specs.powerset else model

    def _load_embedding(self):
        name = self.settings.embedding
        try:
            from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
            import torch
            emb = PretrainedSpeakerEmbedding(name, device=torch.device(self.settings.device))
        except Exception as problem:
            raise ModelAccessError("Could not load %s: %s" % (name, type(problem).__name__)) from None
        if name.startswith("speechbrain/") and type(emb).__name__ != "SpeechBrainPretrainedSpeakerEmbedding":
            raise ModelAccessError("%s did not load as a SpeechBrain model (%s)" % (name, type(emb).__name__))
        self.specs["embedding_dimension"] = int(emb.dimension)
        self.specs["embedding_class"] = type(emb).__name__
        return emb

    def _warm_up(self):
        """One forward pass so the first conversation does not pay for it."""
        import torch
        n = int(self.duration * self.settings.sample_rate)
        with INFERENCE_LOCK, torch.no_grad():
            seg = self.config.segmentation(torch.zeros(1, 1, n))
            self.specs["frames_per_chunk"] = int(seg.shape[1])
            self.specs["local_speakers"] = int(seg.shape[2])

    def ready(self):
        return self.status == "ready"

    def summary(self):
        return {"status": self.status, "error": self.error, "startup_seconds": self.startup_seconds,
                "segmentation": self.settings.segmentation, "embedding": self.settings.embedding, "specs": self.specs}


def wearer_status(similarity, clean_seconds, settings, has_profile):
    if not has_profile:
        return "unknown"
    if clean_seconds < settings.wearer_min_seconds or similarity is None:
        return "pending"
    if similarity >= settings.wearer_yes:
        return "wearer"
    if similarity <= settings.wearer_no:
        return "other"
    return "uncertain"


class Session:
    """One conversation. feed() with 16 kHz mono float32 audio as it arrives; process() runs the pipeline over
    every complete step and calls on_event(dict) for each event. A worker thread can call process() in a loop."""

    def __init__(self, models, session_id, settings, profile=None, on_event=None, clock=time.monotonic):
        from diart import SpeakerDiarization
        self.models, self.settings, self.session_id = models, settings, session_id
        self.on_event = on_event or (lambda event: None)
        self.clock = clock
        self.profile = None
        if profile is not None:
            v = np.asarray(profile["embedding"], dtype=np.float64)
            self.profile = v / np.linalg.norm(v)
        self.sr = settings.sample_rate
        self.duration, self.step = models.duration, settings.step
        self.latency = models.config.latency
        frames = models.specs.get("frames_per_chunk") or 589      # segmentation-3.0: 589 frames per 10 s
        self.frame = self.duration / frames
        self.generation = next(_GENERATIONS)
        self.pipeline = SpeakerDiarization(models.config)   # its own clustering state; the model weights are shared
        self._watch_segmentation()
        self._lock = threading.Lock()
        self._processing = threading.Lock()                  # one process() at a time (worker or final flush)
        self._reset_audio(0.0)
        self.received = 0.0                                  # seconds of band audio received (session clock)
        self.arrivals = []                                   # (session time, wall time) at the end of each feed
        self.speakers = {}                                   # "s0" -> {"wearer", "similarity", "clean_seconds", "first"}
        self.revision = 0
        self.available = True
        self.stats = {"steps": 0, "step_seconds": [], "lag": [], "gaps": []}

    def _watch_segmentation(self):
        """Keep the raw segmentation of each chunk: it shows two voices at once even when diart cannot identify
        one of them (no clean speech of theirs in the window), so overlap never depends on identification."""
        block = self.pipeline.segmentation
        self._last_seg = None

        def watched(batch):
            out = block(batch)
            self._last_seg = out
            return out
        self.pipeline.segmentation = watched

    def _raw_overlap(self, end, a, b):
        """Intervals inside [a, b) where the segmentation hears two or more voices at once."""
        if self._last_seg is None:
            return []
        seg = self._last_seg[0].detach().cpu().numpy() if hasattr(self._last_seg, "detach") else np.asarray(self._last_seg[0])
        res = self.duration / seg.shape[0]
        start = end - self.duration
        busy = (seg > self.settings.tau_active).sum(axis=1) >= 2
        out = []
        for k in np.flatnonzero(busy):
            x, y = start + k * res, start + (k + 1) * res
            if y <= a or x >= b:
                continue
            x, y = max(a, x), min(b, y)
            if out and x - out[-1][1] <= res + 1e-6:
                out[-1][1] = y
            else:
                out.append([x, y])
        return [(x, y) for x, y in out if y - x >= res]

    def _reset_audio(self, at):
        pad = self.duration - self.step                      # leading silence: the first output after one step
        self._audio = np.zeros(int(round(pad * self.sr)), dtype=np.float32)
        self._audio_start = at - pad                         # session time of self._audio[0]
        self.next_end = at + self.step

    # ------------------------------------------------------------------ audio in (any thread)

    def feed(self, samples, start=None):
        """samples: float32 at 16 kHz. start: session time of the first sample (None = right after the last)."""
        samples = np.asarray(samples, dtype=np.float32)
        with self._lock:
            if start is not None and start > self.received + 0.05:
                self._gap(self.received, start, "no audio received (reconnect?)")
                self._audio = np.concatenate([self._audio, np.zeros(int(round((start - self.received) * self.sr)), np.float32)])
                self.received = start
            elif start is not None and start < self.received - 0.01:   # already have it: drop the repeat
                skip = int(round((self.received - start) * self.sr))
                samples = samples[skip:]
            self._audio = np.concatenate([self._audio, samples])
            self.received += len(samples) / self.sr
            self.arrivals.append((self.received, self.clock()))
            self.arrivals = self.arrivals[-400:]

    def backlog(self):
        """Seconds of received audio not yet diarized."""
        return max(0.0, self.received - (self.next_end - self.step))

    def _arrival(self, t):
        """Wall-clock time the audio at session time t arrived (None if not yet)."""
        return next((w for at, w in self.arrivals if at >= t - 1e-6), None)

    def wall_lag(self):
        """How long ago the oldest audio still waiting for diarization arrived. Grows only when processing is
        too slow, not when a burst of audio arrives at once (a replay, or a reconnect flushing its queue)."""
        arrived = self._arrival(self.next_end - self.step + 1e-3)
        return 0.0 if arrived is None else max(0.0, self.clock() - arrived)

    # ------------------------------------------------------------------ processing (worker thread)

    def process(self):
        """Diarize every complete step that has arrived. Returns the number of steps done."""
        with self._processing:
            return self._process()

    def _process(self):
        done = 0
        while True:
            with self._lock:
                if self.wall_lag() > self.settings.max_lag:
                    self._overload()
                    continue
                end = self.next_end
                a = int(round((end - self.duration - self._audio_start) * self.sr))
                b = a + int(round(self.duration * self.sr))
                if b > len(self._audio):
                    return done
                chunk = self._audio[a:b].copy()
                arrived = next((w for t, w in self.arrivals if t >= end - 1e-6), self.clock())
            self._step(chunk, end, arrived)
            done += 1
            with self._lock:
                self.next_end = round(end + self.step, 6)
                keep = int(round((self.next_end - self.duration - self._audio_start) * self.sr))
                if keep > 0:                                  # keep only what the next chunk needs
                    self._audio = self._audio[keep:]
                    self._audio_start += keep / self.sr

    def _step(self, chunk, end, arrived):
        import torch
        from pyannote.core import SlidingWindow, SlidingWindowFeature
        started = self.clock()
        wav = SlidingWindowFeature(chunk.reshape(-1, 1), SlidingWindow(start=end - self.duration, duration=1 / self.sr,
                                                                        step=1 / self.sr))
        with INFERENCE_LOCK, torch.no_grad():
            annotation, region_audio = self.pipeline([wav])[0]
        region = region_audio.extent
        took = self.clock() - started
        lag = self.clock() - arrived
        self.stats["steps"] += 1
        self.stats["step_seconds"].append(round(took, 4))
        self.stats["lag"].append(round(lag, 3))
        if not self.available and lag <= self.settings.max_lag / 2:
            self.available = True
            self._emit({"type": "status", "attribution": "available", "reason": "caught up"})
        a, b = max(0.0, region.start), region.end
        if b <= 0:
            return
        segments = self._partition(annotation, a, b)
        for x, y in self._raw_overlap(end, a, b):              # overlap the identification could not see
            for seg in segments:
                if min(y, seg["end"]) - max(x, seg["start"]) >= 0.5 * (seg["end"] - seg["start"]) and len(seg["speakers"]) < 2:
                    seg["overlap"], seg["unidentified"] = True, True
        for seg in segments:                                  # clean single-speaker time, for wearer decisions
            if len(seg["speakers"]) == 1:
                s = self.speakers.setdefault(seg["speakers"][0], {"wearer": None, "similarity": None,
                                                                  "clean_seconds": 0.0, "first": seg["start"]})
                s["clean_seconds"] += seg["end"] - seg["start"]
        self._match_wearer()
        for seg in segments:
            seg["wearer"] = self._segment_wearer(seg["speakers"], seg.get("unidentified", False))
        self._emit({"type": "speaker", "start": round(a, 3), "end": round(b, 3), "segments": segments,
                    "processing_lag": round(lag, 3), "algorithmic_latency": self.latency,
                    "attribution": "available" if self.available else "unavailable"})

    def _partition(self, annotation, a, b):
        """The region [a, b) cut wherever the set of active speakers changes; every moment is covered.
        diart's segments sit on the segmentation frame grid, about a frame off the region's edges: edges within
        two frames of a region boundary are snapped to it (otherwise every region starts with a false
        "no speech" sliver)."""
        tol = 2 * self.frame
        tracks = []
        for seg, _, label in annotation.itertracks(yield_label=True):
            start, end = seg.start, seg.end
            if abs(start - a) <= tol:
                start = a
            if end >= b - tol and end > a:
                end = b
            if end > a and start < b:
                tracks.append((max(a, start), min(b, end), "s" + str(label).replace("speaker", "")))
        cuts = sorted({a, b} | {t for s, e, _ in tracks for t in (s, e)})
        out = []
        for x, y in zip(cuts, cuts[1:]):
            if y - x < 1e-4:
                continue
            active = sorted({spk for s, e, spk in tracks if s < y and e > x})
            if out and out[-1]["speakers"] == active:
                out[-1]["end"] = round(y, 3)
            else:
                out.append({"start": round(x, 3), "end": round(y, 3), "speakers": active, "overlap": len(active) > 1})
        return out

    def _match_wearer(self):
        clustering = self.pipeline.clustering
        if clustering.centers is None:
            return
        changed = []
        for c in sorted(clustering.active_centers):
            label = "s%d" % c
            s = self.speakers.setdefault(label, {"wearer": None, "similarity": None, "clean_seconds": 0.0, "first": None})
            if self.profile is not None:
                v = np.asarray(clustering.centers[c], dtype=np.float64)
                n = np.linalg.norm(v)
                s["similarity"] = round(float(v @ self.profile / n), 4) if n else None
            status = wearer_status(s["similarity"], s["clean_seconds"], self.settings, self.profile is not None)
            if status != s["wearer"]:
                changed.append((label, s["wearer"], status))
                s["wearer"] = status
        for label, before, after in changed:
            self.revision += 1
            self._emit({"type": "revision", "speaker": label, "wearer": after, "previous": before,
                        "similarity": self.speakers[label]["similarity"],
                        "clean_seconds": round(self.speakers[label]["clean_seconds"], 2),
                        "applies_from": self.speakers[label]["first"]})

    def _segment_wearer(self, active, unidentified=False):
        if not active:
            return "uncertain" if unidentified else "none"
        states = [self.speakers.get(s, {}).get("wearer") for s in active]
        if "wearer" in states:
            return "wearer+other" if len(active) > 1 or unidentified else "wearer"
        if unidentified:
            return "uncertain"                                # someone unidentified also talks: could be the wearer
        if all(st == "other" for st in states):
            return "other"
        return "uncertain"

    def _gap(self, a, b, reason):
        self.stats["gaps"].append((round(a, 3), round(b, 3), reason))
        self._emit({"type": "gap", "start": round(a, 3), "end": round(b, 3), "reason": reason})

    def _overload(self):
        """Too far behind: say so, skip the backlog (marked as a gap) and start a fresh diarization state."""
        from diart import SpeakerDiarization
        skipped_from = self.next_end - self.step
        self.available = False
        self._emit({"type": "status", "attribution": "unavailable",
                    "reason": "speaker processing fell %.1f s behind; skipping ahead" % self.wall_lag()})
        self._gap(skipped_from, self.received, "skipped: processing overloaded")
        self.pipeline = SpeakerDiarization(self.models.config)
        self.generation = next(_GENERATIONS)                  # session speaker ids restart: a new generation
        self.speakers = {}
        self._reset_audio(self.received)
        self._emit({"type": "generation", "generation": self.generation, "from": round(self.received, 3)})

    def _emit(self, event):
        event.update(session_id=self.session_id, generation=self.generation, revision=self.revision)
        self.on_event(event)

    def summary(self):
        steps = sorted(self.stats["step_seconds"])
        lags = sorted(self.stats["lag"])
        pct = lambda xs, q: xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None
        return {"session_id": self.session_id, "generation": self.generation, "received_seconds": round(self.received, 2),
                "steps": self.stats["steps"], "step_seconds_median": pct(steps, 0.5), "step_seconds_p95": pct(steps, 0.95),
                "processing_lag_median": pct(lags, 0.5), "processing_lag_p95": pct(lags, 0.95),
                "algorithmic_latency": self.latency, "gaps": self.stats["gaps"],
                "speakers": {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                             for k, v in self.speakers.items()}}


class Worker:
    """Runs a session's processing on its own thread so feeding never waits for the models."""

    def __init__(self, session, poll=0.05):
        self.session, self.poll = session, poll
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="diart-%s" % session.session_id, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                if not self.session.process():
                    self._stop.wait(self.poll)
            except Exception as problem:                     # never take transcription down with us
                self.session.available = False
                self.session._emit({"type": "status", "attribution": "unavailable",
                                    "reason": "speaker processing error: %s" % type(problem).__name__})
                self._stop.wait(1.0)

    def stop(self, flush=True):
        if flush:
            try:
                self.session.process()
            except Exception:
                pass
        self._stop.set()
