"""STAND-IN models for testing the service's MECHANICS (timeline, events, state, protocol) with diart's real
clustering and aggregation. Tones stand in for voices: these tests say nothing about real-audio accuracy."""
import numpy as np
import torch

SR = 16000
VOICES = {"A": 440.0, "B": 1800.0, "C": 900.0}          # A is the wearer in the tests


def tone(freq, seconds, amp=0.2):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


class Specs:
    def __init__(self, duration):
        self.duration = duration


class FakeSegmentation(torch.nn.Module):
    """(batch, 1, samples) -> (batch, frames, 3): local speaker k is active where tone VOICES[k] is loud."""
    frames_per_second = 20

    def forward(self, waveform):
        batch, _, n = waveform.shape
        hop = SR // self.frames_per_second
        frames = n // hop
        out = torch.zeros(batch, frames, 3)
        x = waveform[:, 0, :frames * hop].reshape(batch, frames, hop).numpy()
        spectrum = np.abs(np.fft.rfft(x, axis=2))
        freqs = np.fft.rfftfreq(hop, 1 / SR)
        for k, f in enumerate(VOICES.values()):
            band = (freqs > f - 60) & (freqs < f + 60)
            energy = spectrum[:, :, band].sum(axis=2)
            out[:, :, k] = torch.from_numpy((energy > 50.0).astype(np.float32))
        return out


class FakeEmbedding:
    """(batch, 1, samples), masks (batch, frames) -> (batch, 32): band energies of the masked audio."""
    dimension = 32

    def to(self, device):
        return self

    def __call__(self, waveforms, masks=None):
        x = waveforms[:, 0, :].numpy()
        out = np.full((x.shape[0], self.dimension), np.nan)
        for i in range(x.shape[0]):
            sig = x[i]
            if masks is not None:
                m = torch.nn.functional.interpolate(masks[i][None, None].float(), size=len(sig), mode="nearest")[0, 0]
                sig = sig[(m > 0.5).numpy()]
            if len(sig) < 1600:
                continue
            spec = np.abs(np.fft.rfft(sig[:16000]))
            bands = np.array([b.sum() for b in np.array_split(spec, self.dimension)])
            if bands.sum() > 0:
                out[i] = bands / np.linalg.norm(bands)
        return out


def fake_models(settings, duration=5.0):
    from speaker_service.engine import Models
    m = Models(settings, segmentation_loader=lambda: FakeSegmentation(), embedding_loader=lambda: FakeEmbedding())
    m.specs = {"duration": duration}
    return m.load()


def profile_for(freq):
    emb = FakeEmbedding()(torch.from_numpy(tone(freq, 2.0))[None, None], None)[0]
    return {"embedding": (emb / np.linalg.norm(emb)).tolist(), "embedding_model": "fake"}
