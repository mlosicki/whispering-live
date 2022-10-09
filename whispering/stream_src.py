import sys
from logging import getLogger
from subprocess import Popen
from typing import Optional

import ffmpeg
import numpy as np
from whisper.audio import SAMPLE_RATE

from whispering.deepl_cli import DeeplCli
from whispering.schema import Context
from whispering.transcriber import WhisperStreamingTranscriber

logger = getLogger(__name__)

external_translator = None


def open_stream(*, stream: Optional[str], direct_url: Optional[str]) -> Popen:
    logger.info("Opening stream...")
    if not direct_url:
        import streamlink

        stream_options = streamlink.streams(stream)
        if not stream_options:
            raise RuntimeError(f"No playable streams found on this URL: {stream}")

        # TODO maybe there's an streamlink API to do it properly
        if "audio_only" in stream_options:  # Twitch
            stream = stream_options["audio_only"].url
        elif "audio_opus" in stream_options:  # YT
            stream = stream_options["audio_opus"].url
        elif "audio_mp4a" in stream_options:  # YT
            stream = stream_options["audio_mp4a"].url
        elif "best" in stream_options:  # Twitch
            stream = stream_options["best"].url
        else:
            stream = next(iter(stream_options.values())).url
    else:
        stream = direct_url

    logger.debug(f"Stream input: {stream}")
    try:
        process = (
            ffmpeg.input(stream, loglevel="panic")
            .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE)
            .global_args("-re")  # Argument to act as a live stream
            .run_async(pipe_stdout=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return process


def transcribe_from_stream(
    *,
    wsp: WhisperStreamingTranscriber,
    interval: int,
    stream: str,
    direct_url: str,
    ctx: Context,
    no_progress: bool,
    external_translator_name: str,
) -> None:
    logger.info("Ready to transcribe")
    idx: int = 0
    process = open_stream(stream=stream, direct_url=direct_url)

    read_buffer = interval * SAMPLE_RATE * 2

    while process.poll() is None:
        logger.debug(f"Audio #: {idx}")
        assert process.stdout is not None

        # TODO this function is very similar to the function from upstream
        # and could be generalized to only read from a queue of np.ndarrays
        # The source (ffmpeg, mic, etc) should not matter
        in_bytes = process.stdout.read(read_buffer)
        if not in_bytes:
            break
        audio = np.frombuffer(in_bytes, np.int16).flatten().astype(np.float32) / 32768.0

        logger.debug(f"Got {len(in_bytes)} bytes")
        if not no_progress:
            sys.stderr.write("Analyzing")
            sys.stderr.flush()
        full_text = ""
        for chunk in wsp.transcribe(audio=audio, ctx=ctx):
            if not no_progress:
                sys.stderr.write("\r")
                sys.stderr.flush()
            print(f"{chunk.start:.2f}->{chunk.end:.2f}\t{chunk.text}", flush=True)
            full_text += f"{chunk.text}\n\n"
            if not no_progress:
                sys.stderr.write("Analyzing")
                sys.stderr.flush()
        if external_translator_name == "deepl":
            # TODO refactor as class
            global external_translator
            if external_translator is None:
                external_translator = DeeplCli()
            translated = external_translator.translate(text=full_text)
            if translated:
                print("Translation:", flush=True)
                print(f"{translated}", flush=True)
        idx += 1
        if not no_progress:
            sys.stderr.write("\r")
            sys.stderr.flush()
    logger.info("Stream ended")
    process.wait()
