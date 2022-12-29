from typing import Optional
from jina import Executor, requests, Document, DocumentArray
from jina.logging.logger import JinaLogger
import torch
import tempfile

# FIXME: this is a workaround for the issue of executor not being able to load the local whisper module
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import whisper


class WhisperExecutor(Executor):
    def __init__(
        self,
        name: str = 'medium',
        device: Optional[str] = None,
        mode: str = 'encoder-decoder',
        **kwargs,
    ):
        super().__init__()

        self.logger = JinaLogger(self.__class__.__name__)

        if not device:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device

        self._mode = mode

        self._model = whisper.load_model(name, device=device)
        self._model.eval()

        self._options = whisper.DecodingOptions(
            fp16=False if self._device == 'cpu' else True
        )

        self.logger.info(
            f'Loaded {"multilingual" if self._model.is_multilingual else "english-only"} model {name}'
        )

        # options = dict(beam_size=beam_size, best_of=best_of)
        # self._transcribe_options = dict(task="transcribe", **options)
        # self._translate_options = dict(task="translate", **options)

    def load_audio(self, docs):
        # load audio into tensors format
        for doc in docs:
            if doc.uri and doc.uri.startswith('http'):
                audio = whisper.load_audio(doc.uri)
            elif doc.blob is not None:
                ext = doc.tags.get('ext', 'wav')
                with tempfile.NamedTemporaryFile(
                    suffix=f'.{ext}' if docs.tags.get('ext', None) else None
                ) as fp:
                    fp.write(doc.blob)
                    fp.flush()
                    audio = whisper.load_audio(fp.name)
            else:
                raise ValueError('No audio content found in the document')
            audio = whisper.pad_or_trim(audio)
            doc.tensor = whisper.log_mel_spectrogram(audio)

    @requests
    def transcribe(self, docs, **kwargs):
        # in encoder mode, only the encoder is used to extract the audio features
        if self._mode.lower() == 'encoder':
            self.load_audio(docs)
            with torch.inference_mode():
                docs.embeddings = self._model.encoder(docs.tensors)
        # in decoder mode, only the decoder is used to transcribe the audio features
        elif self._mode.lower() == 'decoder':
            if docs.embeddings is None:
                raise ValueError(
                    'No audio embeddings found in the document, '
                    'please use encoder mode to extract the embeddings first'
                )
            with torch.inference_mode():
                result = whisper.decoding.decode(
                    self._model, docs.embeddings, options=self._options
                )
                docs.texts = [r.text for r in result]
        else:
            self.load_audio(docs)
            with torch.inference_mode():
                result = whisper.decoding.decode(
                    self._model, docs.tensors, options=self._options
                )
                docs.texts = [r.text for r in result]
