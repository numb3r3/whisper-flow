

**Whisper-Flow** is an efficient and scalable deployment of [OpenAI Whisper](https://github.com/openai/whisper) model. 
Whisper is a general-purpose speech recognition model that can be used for a variety of tasks. 
It is a transformer encoder-decoder model that the encoder and decoder can be separately hosted on different devices.
What's more, encoder and decoder have different throughput, which makes it possible to deploy encoder on a high-performance device and decoder on a low-power device.


# 🚀 Quickstart

## 💾 Installation

```bash
pip install -r requirments.txt
```

---

## 🚌 Usage


### Start the server 

```python
from jina import Flow

with Flow(port=52000).add(
    name='encoder', uses='config.yml', uses_with={'mode': 'encoder'}
).add(name='decoder', uses='config.yml', uses_with={'mode': 'decoder'}) as f:
    f.block()
```


### Use the client

```python
from jina import Client, DocumentArray, Document

docs = DocumentArray.empty(1)
docs[0].uri = 'audio.flac'
docs[0].load_uri_to_blob()
docs[0].tags['ext'] = 'flac'

client = Client(port=52000)
result = client.post(on='/', inputs=docs, return_results=True)
print(result.texts)
```