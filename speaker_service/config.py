"""Speaker service settings, all from the environment (or the repository's git-ignored .env). No secrets in code.

HF_TOKEN                      Hugging Face token with access to the gated pyannote/segmentation-3.0 (required)
SPEAKER_SEGMENTATION_MODEL    default pyannote/segmentation-3.0
SPEAKER_EMBEDDING_MODEL       default speechbrain/spkrec-ecapa-voxceleb
SPEAKER_STEP                  seconds between diarization outputs (default 0.5)
SPEAKER_LATENCY               seconds of look-ahead before a region is final (default 3.0; step..duration)
SPEAKER_TAU_ACTIVE / SPEAKER_RHO_UPDATE / SPEAKER_DELTA_NEW / SPEAKER_MAX_SPEAKERS   diart clustering
SPEAKER_WEARER_YES / SPEAKER_WEARER_NO     cosine similarity to the enrolled wearer: at or above YES is the
                              wearer, at or below NO is not, in between is uncertain (calibrate: replay --calibrate)
SPEAKER_WEARER_MIN_SECONDS    clean speech a session speaker needs before a wearer decision (default 2.0)
SPEAKER_MAX_LAG               seconds of unprocessed audio before attribution is marked unavailable (default 6)
SPEAKER_PROFILE               where the enrolled wearer profile is kept (default profiles/wearer_diart.json)
SPEAKER_DEVICE                cpu (default) or mps / cuda
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dotenv():
    values = {}
    try:
        with open(os.path.join(REPO, ".env"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:]
                key, sep, value = line.partition("=")
                if sep and key.strip() and not key.strip().startswith("#"):
                    values[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        pass
    return values


_FILE = _dotenv()


def get(name, default=None):
    value = os.environ.get(name)
    if value is None or value == "":
        value = _FILE.get(name, default)
    return value


def number(name, default):
    try:
        return float(get(name, default))
    except (TypeError, ValueError):
        return float(default)


class Settings:
    def __init__(self):
        self.hf_token = get("HF_TOKEN") or None
        self.segmentation = get("SPEAKER_SEGMENTATION_MODEL", "pyannote/segmentation-3.0")
        self.embedding = get("SPEAKER_EMBEDDING_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
        self.step = number("SPEAKER_STEP", 0.5)
        self.latency = number("SPEAKER_LATENCY", 3.0)       # measured: 2 s missed 21% of speech, 3 s 16%, 4 s 8%
        self.tau_active = number("SPEAKER_TAU_ACTIVE", 0.4)
        self.rho_update = number("SPEAKER_RHO_UPDATE", 0.3)
        self.delta_new = number("SPEAKER_DELTA_NEW", 1.0)
        self.max_speakers = int(number("SPEAKER_MAX_SPEAKERS", 10))
        self.gamma = number("SPEAKER_GAMMA", 3)
        self.beta = number("SPEAKER_BETA", 10)
        self.wearer_yes = number("SPEAKER_WEARER_YES", 0.45)
        self.wearer_no = number("SPEAKER_WEARER_NO", 0.25)
        self.wearer_min_seconds = number("SPEAKER_WEARER_MIN_SECONDS", 2.0)
        self.max_lag = number("SPEAKER_MAX_LAG", 6.0)
        self.profile = get("SPEAKER_PROFILE", os.path.join(REPO, "profiles", "wearer_diart.json"))
        self.device = get("SPEAKER_DEVICE", "cpu")
        self.sample_rate = 16000

    def public(self):
        """For /ready and logs: everything except the token."""
        out = {k: v for k, v in vars(self).items() if k != "hf_token"}
        out["hf_token"] = "set" if self.hf_token else "missing"
        return out
