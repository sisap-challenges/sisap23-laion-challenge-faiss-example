import argparse
import faiss
import h5py
import numpy as np
from sklearn import preprocessing
import os
from pathlib import Path
from urllib.request import urlretrieve
import time


def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)


def prepare(kind, size):
    url = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge"
    # url = "http://ingeotec.mx/~sadit/metric-datasets/LAION/SISAP23-Challenge"
    task = {
        "query": f"{url}/public-queries-10k-{kind}.h5",
        "dataset": f"{url}/laion2B-en-{kind}-n={size}.h5",
    }

    for version, url in task.items():
        download(url, os.path.join("data", kind, size, f"{version}.h5"))


def load_clip768(dbname, key):
    with h5py.File(dbname, 'r') as f:
        X = np.array(f[key], dtype=np.float32)
        X = preprocessing.normalize(X, norm='l2')

    return X


def store_results(dst, algo, kind, D, I, buildtime, querytime, params, size):
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, 'w')
    f.attrs['algo'] = algo
    f.attrs['data'] = kind
    f.attrs['buildtime'] = buildtime
    f.attrs['querytime'] = querytime
    f.attrs['size'] = size
    f.attrs['params'] = params
    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()


def create_hnsw(X, nsize, dim, neighborhoodsize, efConstruction):
    starttime = time.time()
    index = faiss.IndexHNSWFlat(dim, neighborhoodsize, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = efConstruction
    index.add(X)
    buildtime = time.time() - starttime
    name = f"faiss HSNW d={dim} M={neighborhoodsize} efC={efConstruction}"
    mem = 0  # avoids saving the index for now
    return index, name, mem, buildtime


def run_hsnw(kind, key, size, k=30):
    print("Running", kind)
    prepare(kind, size)

    datafile = os.path.join("data", kind, size, "dataset.h5")
    queriesfile = os.path.join("data", kind, size, "query.h5")
    data = load_clip768(datafile, key)
    queries = load_clip768(queriesfile, key)
    n, d = data.shape
    index, name, mem, buildtime = create_hnsw(data, n, d, 32, 500)
    assert index.is_trained

    for efSearch in [32, 64, 128, 256, 512]:
        print(f"Starting HNSW search on {queries.shape} with efS={efSearch}")
        start = time.time()
        index.hnsw.efSearch = efSearch
        D, I = index.search(queries, k)
        elapsed_search = time.time() - start
        print(f"Done searching HNSW in {elapsed_search}s.")

        I = I + 1  # FAISS is 0-indexed, groundtruth is 1-indexed

        identifier = f"{name} efSearch={efSearch}"
        store_results(os.path.join("result/", kind, size, f"{identifier}.h5"), "faissHNSW", kind, D, I, buildtime, elapsed_search, identifier, size)


def run_ivf(kind, key, size, k=30):
    print("Running", kind)
    prepare(kind, size)

    nlist = 128  # number of clusters/centroids to build the IVF from
    datafile = os.path.join("data", kind, size, "dataset.h5")
    queriesfile = os.path.join("data", kind, size, "query.h5")

    if kind.startswith("pca"):
        data = np.array(h5py.File(datafile, "r")[key])
        queries = np.array(h5py.File(queriesfile, "r")[key])
        n, d = data.shape
        index_identifier = f"IVF{nlist},Flat"
        index = faiss.index_factory(d, index_identifier)
    elif kind.startswith("hamming"):
        data = np.array(h5py.File(datafile, "r")[key])
        queries = np.array(h5py.File(queriesfile, "r")[key])
        n, d = data.shape
        index_identifier = f"BIVF{nlist},Flat"  # use binary IVF index
        d = 64 * d  # one chunk contains 64 bits
        index = faiss.index_binary_factory(d, index_identifier)
        # create view to interpret original uint64 as 8 chunks of uint8
        data = np.array(data).view(dtype="uint8")
        queries = np.array(queries).view(dtype="uint8")
    elif kind.startswith("clip768"):
        data = load_clip768(datafile, key)
        queries = load_clip768(queriesfile, key)
        n, d = data.shape
        index_identifier = f"IVF{nlist},Flat"
        index = faiss.index_factory(d, index_identifier, faiss.METRIC_INNER_PRODUCT)

    print(f"Training index on {data.shape}")
    start = time.time()
    index.train(data)
    index.add(data)
    elapsed_build = time.time() - start
    print(f"Done training in {elapsed_build}s.")
    assert index.is_trained

    for nprobe in [1, 2, 5, 10, 20, 50, 100]:
        print(f"Starting search on {queries.shape} with nprobe={nprobe}")
        start = time.time()
        index.nprobe = nprobe
        D, I = index.search(queries, k)
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        I = I + 1  # FAISS is 0-indexed, groundtruth is 1-indexed

        identifier = f"index=({index_identifier}),query=(nprobe={nprobe})"

        store_results(os.path.join("result/", kind, size, f"{identifier}.h5"), "faissIVF", kind, D, I, elapsed_build, elapsed_search, identifier, size)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        default="300K"
    )
    parser.add_argument(
        "--k",
        default=30,
    )

    args = parser.parse_args()

    assert args.size in ["100K", "300K", "10M", "30M", "100M"]

    # run("pca32v2", "pca32", args.size, args.k)
    # run("pca96v2", "pca96", args.size, args.k)
    # run("hammingv2", "hamming", args.size, args.k)
    run_hsnw("clip768v2", "emb", args.size, args.k)

