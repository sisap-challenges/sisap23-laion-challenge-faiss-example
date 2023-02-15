import faiss
import h5py
import numpy as np
import os
from pathlib import Path
from urllib.request import urlretrieve
import time 

def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)

def prepare(kind):
    url = "http://ingeotec.mx/~sadit/metric-datasets/LAION/SISAP23-Challenge"

    task = {
        "query": f"{url}/{kind}/en-queries/public-queries-10k-{kind}.h5",
        "dataset": f"{url}/{kind}/en-bundles/laion2B-en-{kind}-n=100K.h5",
        # "groundtruth": f"{url}/public-queries/en-gold-standard-public/small-laion2B-en-public-gold-standard-100K.h5",
    }

    for version, url in task.items():
        download(url, os.path.join("data", kind, f"{version}.h5"))

def store_results(dst, kind, D, I, buildtime, querytime, params):
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['kind'] = kind
    f.attrs['build'] = buildtime
    f.attrs['querytime'] = querytime
    f.attrs['params'] = params
    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()

def run(kind, k=30):
    print("Running", kind)
    
    prepare(kind)

    data = np.array(h5py.File(os.path.join("data", kind, "dataset.h5"), "r")[kind])
    queries = np.array(h5py.File(os.path.join("data", kind, "query.h5"), "r")[kind])
    n, d = data.shape

    nlist = 128 # number of clusters/centroids to build the IVF from

    if kind == "pca32" or kind == "pca96":
        index_identifier = f"IVF{nlist},Flat"
        index = faiss.index_factory(d, index_identifier)
    elif kind == "hamming":
        index_identifier = f"BIVF{nlist},Flat" # use binary IVF index
        d = 64 * d # one chunk contains 64 bits
        index = faiss.index_binary_factory(d, index_identifier)
        # create view to interpret original uint64 as 8 chunks of uint8
        data = np.array(data).view(dtype="uint8")
        queries = np.array(queries).view(dtype="uint8")
    else:
        raise Exception(f"unsupported input type {kind}")

    print(f"Training index on {data.shape}")
    start = time.time()
    index.train(data)
    elapsed_build = time.time() - start
    print(f"Done training in {elapsed_build}s.")

    for nprobe in [1, 2, 5, 10, 50]:
        print(f"Starting search on {queries.shape} with nprobe={nprobe}")
        start = time.time()
        index.nprobe = nprobe
        D, I = index.search(queries, k)
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        identifier = f"index=({index_identifier}), query=(nprobe={nprobe})"

        store_results(os.path.join("result/", kind, f"{identifier}.h5"), kind, D, I, elapsed_build, elapsed_search, identifier)

if __name__ == "__main__":
    run("pca32")
    run("pca96")
    run("hamming")