from __future__ import annotations

import base64
import io
import uuid
import wave
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openai.types.realtime.realtime_response_create_params import RealtimeResponseCreateParams


def response_wants_audio(response: RealtimeResponseCreateParams | None) -> bool:
    """Whether a response should produce audio (and audio events) vs. text only.

    Mirrors the OpenAI realtime semantics for ``output_modalities``: an absent
    value (``None``) or an empty list, or an explicit ``"audio"`` entry means
    audio; a non-empty list without ``"audio"`` (e.g. ``["text"]``) means text
    only.
    """
    if response is None:
        return True
    mods = response.output_modalities
    return not mods or "audio" in mods


def audio_to_wav_base64(audio: np.ndarray, sample_rate: int) -> str:
    """Encode a mono 16-bit WAV payload without touching the filesystem."""
    audio_array = np.asarray(audio)
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=1)
    if np.issubdtype(audio_array.dtype, np.floating):
        pcm = (np.clip(audio_array, -1.0, 1.0) * 32767.0).astype("<i2")
    else:
        pcm = np.clip(audio_array, -32768, 32767).astype("<i2")

    with io.BytesIO() as wav_io:
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
        return base64.b64encode(wav_io.getvalue()).decode("ascii")


def is_out_of_band(response: RealtimeResponseCreateParams | None) -> bool:
    """Whether a response is *out-of-band* (``conversation="none"``).

    Out-of-band responses are generated against a temporary context and are
    never threaded into the default conversation: their ``input`` (if any)
    seeds a throwaway chat, their assistant output is not committed back, and
    their ``conversation_id`` is reported as ``null``. Any other ``conversation``
    value (``"auto"``, ``None``, or an arbitrary id) is treated as in-band.
    """
    return response is not None and response.conversation == "none"


def next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def int2float(sound: np.ndarray) -> np.ndarray:
    """
    Taken from https://github.com/snakers4/silero-vad
    """

    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound
