import faiss
import h5py
import os
from pathlib import Path
from urllib.request import urlretrieve

def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)

def run(kind, k=30):
    url = "http://ingeotec.mx/~sadit/metric-datasets/LAION/SISAP23-Challenge/"

    task = {
        "query": f"{url}/{kind}/en-queries/public-queries-10k-{kind}.h5",
        "dataset": f"{url}/{kind}/en-bundles/laion2B-en-{kind}-n=100K.h5",
        "groundtruth": f"{url}/public-queries/en-gold-standard-public/small-laion2B-en-public-gold-standard-100K.h5",
    }

    for version, url in task.items():
        download(url, os.path.join("data", kind, f"{version}.h5"))

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")
    run("pca32")


