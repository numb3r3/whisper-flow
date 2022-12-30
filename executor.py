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

        self._model = whisper.load_model(name, device='cpu')
        self._model.eval()

        # load partial model to device
        if self._mode == 'encoder':
            self._model.encoder = self._model.encoder.to(self._device)
        elif self._mode == 'decoder':
            self._model.decoder = self._model.decoder.to(self._device)
        else:
            self._model = self._model.to(self._device)

        self._decode_options = dict(fp16=False if self._device == 'cpu' else True)

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
                    suffix=f'.{ext}' if doc.tags.get('ext', None) else None
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
                embeddings = self._model.encoder(docs.tensors.to(self._device))
            docs.embeddings = embeddings.detach().cpu()

        # in decoder mode, only the decoder is used to transcribe the audio features
        elif self._mode.lower() == 'decoder':
            if docs.embeddings is None:
                raise ValueError(
                    'No audio features extracted in the document, '
                    'please use encoder to extract audio features first'
                )
            with torch.inference_mode():
                # get the audio features from the embeddings
                encodings = docs.embeddings.to(self._device)

                # detect the spoken language
                _, probs = self._model.detect_language(encodings)
                lang = max(probs[0], key=probs[0].get)

                # transcribe the audio
                results = whisper.decoding.decode(
                    self._model,
                    encodings,
                    options=whisper.DecodingOptions(
                        **self._decode_options,
                        language=lang,
                        task=kwargs.get('task', 'transcribe'),
                    ),
                )
            for d, r in zip(docs, results):
                d.text = r.text
                d.tags['no_speech_prob'] = r.no_speech_prob
                d.tags['language'] = r.language
        else:
            self.load_audio(docs)
            with torch.inference_mode():
                # get the audio mel-spectrogram from the tensors
                mels = docs.tensors.to(self._device)

                # detect the spoken language
                _, probs = self._model.detect_language(mels)
                lang = max(probs[0], key=probs[0].get)

                # transcribe the audio
                results = whisper.decoding.decode(
                    self._model,
                    mels,
                    options=whisper.DecodingOptions(
                        **self._decode_options,
                        language=lang,
                        task=kwargs.get('task', 'transcribe'),
                    ),
                )

            for d, r in zip(docs, results):
                d.text = r.text
                d.tags['no_speech_prob'] = r.no_speech_prob
                d.tags['language'] = r.language
