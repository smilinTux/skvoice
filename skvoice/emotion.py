"""Lightweight audio emotion detection from raw PCM."""

import struct
import math


def analyze_audio(pcm_bytes: bytes, sample_rate: int = 16000) -> dict:
    """Analyze raw 16-bit signed PCM for basic emotional cues.

    Returns dict with rms_energy, zcr_rate, pitch_hz, and emotion_tags.
    """
    if len(pcm_bytes) < 4:
        return {
            "rms_energy": 0.0,
            "zcr_rate": 0.0,
            "pitch_hz": None,
            "emotion_tags": [],
        }

    # Unpack 16-bit signed little-endian samples
    n_samples = len(pcm_bytes) // 2
    samples = struct.unpack(f"<{n_samples}h", pcm_bytes[: n_samples * 2])

    if not samples:
        return {
            "rms_energy": 0.0,
            "zcr_rate": 0.0,
            "pitch_hz": None,
            "emotion_tags": [],
        }

    # Normalize to [-1, 1]
    norm = [s / 32768.0 for s in samples]

    # RMS energy
    rms = math.sqrt(sum(x * x for x in norm) / len(norm))

    # Zero-crossing rate
    crossings = sum(
        1 for i in range(1, len(norm)) if (norm[i] >= 0) != (norm[i - 1] >= 0)
    )
    zcr = crossings / len(norm) if len(norm) > 1 else 0.0

    # Simple pitch via autocorrelation
    pitch_hz = _estimate_pitch(norm, sample_rate)

    # Derive emotion tags
    tags = _derive_tags(rms, zcr, pitch_hz)

    return {
        "rms_energy": round(rms, 4),
        "zcr_rate": round(zcr, 4),
        "pitch_hz": round(pitch_hz, 1) if pitch_hz else None,
        "emotion_tags": tags,
    }


def _estimate_pitch(samples: list[float], sample_rate: int) -> float | None:
    """Estimate fundamental frequency via autocorrelation.

    Searches for pitch between 70 Hz and 400 Hz.
    """
    min_lag = sample_rate // 400  # 400 Hz upper bound
    max_lag = sample_rate // 70  # 70 Hz lower bound
    n = len(samples)

    if n < max_lag + 1:
        return None

    # Use a subset for speed
    window = min(n, sample_rate)  # max 1 second
    s = samples[:window]

    best_corr = 0.0
    best_lag = 0

    for lag in range(min_lag, min(max_lag + 1, window)):
        corr = sum(s[i] * s[i + lag] for i in range(window - lag))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    if best_lag == 0 or best_corr <= 0:
        return None

    # Check correlation is strong enough relative to zero-lag
    zero_corr = sum(s[i] * s[i] for i in range(window))
    if zero_corr == 0 or best_corr / zero_corr < 0.15:
        return None

    return sample_rate / best_lag


def _derive_tags(
    rms: float, zcr: float, pitch_hz: float | None
) -> list[str]:
    """Map audio features to emotion tags."""
    tags = []

    # Energy
    if rms > 0.08:
        tags.append("energetic")
    elif rms < 0.02:
        tags.append("calm")

    # Speech pace (ZCR as proxy)
    if zcr > 0.15:
        tags.append("rapid_speech")
    elif zcr < 0.05:
        tags.append("slow_speech")

    # Pitch
    if pitch_hz is not None:
        if pitch_hz > 250:
            tags.append("high_pitch")
        elif pitch_hz < 130:
            tags.append("low_pitch")

    return tags


def emotion_context_string(analysis: dict) -> str:
    """Build a human-readable emotion context string for the LLM.

    Returns empty string if no notable emotions detected.
    """
    tags = analysis.get("emotion_tags", [])
    if not tags:
        return ""

    descriptors = []
    for tag in tags:
        match tag:
            case "energetic":
                descriptors.append("energetic")
            case "calm":
                descriptors.append("calm and measured")
            case "rapid_speech":
                descriptors.append("speaking quickly")
            case "slow_speech":
                descriptors.append("speaking slowly")
            case "high_pitch":
                descriptors.append("with high pitch")
            case "low_pitch":
                descriptors.append("with low pitch")

    if not descriptors:
        return ""

    return f"[Speaker sounds {', '.join(descriptors)}]"
