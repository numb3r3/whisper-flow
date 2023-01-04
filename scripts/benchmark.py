from jina import Flow, DocumentArray, Document
import time

# the example input for benchmark, please change the path to your local audio files
docs = DocumentArray.empty(2)
docs[0].uri = 'audio.flac'
docs[0].load_uri_to_blob()

docs[1].uri = 'audio.flac'
docs[1].load_uri_to_blob()

N = 50
MODEL_NAME = 'small'


def doc_gen(num: int = N):
    for _ in range(num):
        yield Document(blob=docs[0].blob)


writer = open('results.txt', 'w')


for n in range(1, 5):
    for m in range(1, 5):
        with Flow().add(
            name='encoder',
            uses='config.yml',
            uses_with={'mode': 'encoder', 'name': MODEL_NAME},
            replicas=n,
        ).add(
            name='decoder',
            uses='config.yml',
            uses_with={'mode': 'decoder', 'name': MODEL_NAME},
            replicas=m,
        ) as f:

            # warm-up
            f.post(on='/', inputs=docs)

            start = time.perf_counter()
            r = f.post(on='/', inputs=doc_gen(), return_results=True, request_size=1)
            elapsed = time.perf_counter() - start
            print(f'{n} encoders x {m} decoders: {elapsed:.2f}s')
            writer.write(f'{n} encoders x {m} decoders: {elapsed:.2f}s\n')
    writer.flush()

writer.close()
