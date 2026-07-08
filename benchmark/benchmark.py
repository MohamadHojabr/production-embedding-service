import time
import psutil
import os


def memory_usage():

    process = psutil.Process(
        os.getpid()
    )

    return (
        process.memory_info()
        .rss
        /
        1024**2
    )


def benchmark(
    service,
    texts,
    batch_size
):

    print(
        f"\nBatch={batch_size}"
    )


    before = memory_usage()


    start=time.time()


    embeddings = service.encode(
        texts,
        batch_size=batch_size
    )


    elapsed=time.time()-start


    after=memory_usage()


    print(
        "Time:",
        elapsed
    )


    print(
        "Memory:",
        after-before,
        "MB"
    )


    print(
        "Shape:",
        embeddings.shape
    )