#!/usr/bin/env python3

from logging import getLogger
from typing import Iterator, Optional

import numpy as np
import torch
from whisper.audio import N_FRAMES, SAMPLE_RATE

from whispering.schema import SpeechSegment

logger = getLogger(__name__)


class VAD:
    def __init__(
        self,
    ):
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
        )
        (
            get_speech_timestamps,
            save_audio,
            _,
            _,
            collect_chunks
        ) = utils
        self.get_speech_timestamps = get_speech_timestamps
        self.collect_chunks = collect_chunks
        self.save_audio = save_audio

    def timestamps(
        self,
        *,
        audio: np.ndarray,
        threshold: float,
    ):
        speech_timestamps = self.get_speech_timestamps(
            torch.from_numpy(audio),
            self.vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=threshold,
            min_silence_duration_ms=2000,
            speech_pad_ms=200
        )
        logger.info(f"Speech timestamps {speech_timestamps}")
        return speech_timestamps

    def __call__(
        self,
        *,
        audio: np.ndarray,
        threshold: float,
        total_block_number: Optional[int] = None,
    ) -> Iterator[SpeechSegment]:
        # audio.shape should be multiple of (N_FRAMES,)

        def my_ret(
            *,
            start_block_idx: int,
            idx: int,
        ) -> SpeechSegment:
            return SpeechSegment(
                start_block_idx=start_block_idx,
                end_block_idx=idx,
                audio=audio[N_FRAMES * start_block_idx : N_FRAMES * idx],
            )

        if total_block_number is None:
            total_block_number = int(audio.shape[0] / N_FRAMES)
        block_unit: int = audio.shape[0] // total_block_number

        start_block_idx = None
        for idx in range(total_block_number):
            start: int = block_unit * idx
            end: int = block_unit * (idx + 1)
            vad_prob = self.vad_model(
                torch.from_numpy(audio[start:end]),
                SAMPLE_RATE,
            ).item()
            if vad_prob > threshold:
                if start_block_idx is None:
                    start_block_idx = idx
            else:
                if start_block_idx is not None:
                    yield my_ret(
                        start_block_idx=start_block_idx,
                        idx=idx,
                    )
                    start_block_idx = None
        if start_block_idx is not None:
            yield my_ret(
                start_block_idx=start_block_idx,
                idx=total_block_number,
            )
