import os
from logging import getLogger
from typing import Optional

import deepl

DEEPL_APIKEY_ENVVAR = "DEEPL_API_KEY"

logger = getLogger(__name__)


# Whisper allows you to also translate the text instead of just transcribing it,
# but the translation results might be better using a dedicated translation service
# This class allows you to translate the provided (transcribed) text via DeepL.
# You need to register for a free API key and provide it via the 'DEEPL_API_KEY' env var
class DeeplCli:
    def __init__(self, *, max_length: int = 300):
        self.translator = deepl.Translator(
            auth_key=os.getenv(DEEPL_APIKEY_ENVVAR),
            server_url="https://api-free.deepl.com",
        )
        self.max_length = max_length

    def translate(self, *, text: str) -> Optional[str]:
        post_text = text[: self.max_length]
        try:
            result = self.translator.translate_text(
                post_text, source_lang="JA", target_lang="EN-US", split_sentences="1"
            )
            return (
                "".join([x.text for x in result])
                if isinstance(result, list)
                else result.text
            )
        except Exception:
            logger.exception("DeepL translation failed")
            return None
