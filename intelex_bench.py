import numpy as np
import pandas as pd
from glob2 import glob
import timeit
from tqdm import tqdm
from itertools import product
from sklearnex import patch_sklearn, set_config, get_config, config_context
import logging

logger = logging.getLogger("sklearnex")
logger.setLevel(logging.INFO)
patch_sklearn()

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC

data = pd.concat([pd.read_csv(item) for item in glob("data/*.csv")]).sample(frac=1)
data = data.drop(columns="timestamp").values
data = data.astype(np.float32)


def do_DBSCAN(X):
    clusters = DBSCAN(eps=3, min_samples=2).fit(X)
    return clusters.labels_


def do_KMeans(X):
    clusters = KMeans(n_clusters=10, n_init=10).fit_transform(X)
    return clusters


def do_RandomForestRegressor(X):
    model = RandomForestRegressor(n_estimators=100).fit(X=X[:, 1:5], y=X[:, 6])
    preds = model.predict(X=X[:, 1:5])
    return preds


def do_LinearRegression(X):
    model = LinearRegression().fit(X=X[:, 1:5], y=X[:, 6])
    preds = model.predict(X=X[:, 1:5])
    return preds


def do_KNeighborsRegressor(X):
    model = KNeighborsRegressor(n_neighbors=11).fit(X=X[:, 1:5], y=X[:, 6])
    preds = model.predict(X=X[:, 1:5])
    return preds


def do_PCA(X):
    dims = PCA().fit_transform(X)
    return dims


def do_SVC_lin(X):
    model = SVC(kernel="linear", probability=True).fit(X=X[:, 1:5], y=X[:, 6])
    preds = model.predict(X=X[:, 1:5])
    return preds


def do_SVC_rbf(X):
    model = SVC(kernel="rbf", probability=True).fit(X=X[:, 1:5], y=X[:, 6])
    preds = model.predict(X=X[:, 1:5])
    return preds


def do_SVC_exp(X):
    model = SVC(kernel="poly", probability=True).fit(X=X[:, 1:5], y=X[:, 6])
    preds = model.predict(X=X[:, 1:5])
    return preds


def do_RandomForestClassification(X):
    model = RandomForestClassifier(n_estimators=100).fit(X=X[:, 1:5], y=X[:, 6])
    preds = model.predict(X=X[:, 1:5])
    return preds


def benchmark(offload: str, searchspace: list, iterations: int = 10):
    bench_results = []
    for algorithm, size in tqdm(searchspace):
        time = timeit.timeit(
            stmt=f"{algorithm}(X=data[:{size},:])", globals=globals(), number=iterations
        )
        bench_results.append(
            {
                "offload": offload,
                "sample_size": size,
                "algorithm": algorithm,
                "time": time,
            }
        )
        pd.Series(
            {
                "offload": offload,
                "sample_size": size,
                "algorithm": algorithm,
                "time": time,
            }
        ).to_csv(f"intermittant_results/{offload}{algorithm}{size}.csv")
    return bench_results


subset_sizes = ["500", "1000", "5000", "10000", "50000", "100000"]
algorithms = [
    "do_RandomForestClassification",
    "do_SVC_rbf",
    "do_SVC_exp",
    "do_SVC_lin",
    "do_PCA",
    "do_KNeighborsRegressor",
    "do_LinearRegression",
    "do_RandomForestRegressor",
    "do_DBSCAN",
    "do_KMeans",
]
algorithms_supported_on_gpu = [
    "do_RandomForestClassification",
    "do_PCA",
    "do_LinearRegression",
    "do_RandomForestRegressor",
    "do_DBSCAN",
]
generic_searchspace = [item for item in product(algorithms, subset_sizes)]
supported_on_gpu_searchspace = [
    item for item in product(algorithms_supported_on_gpu, subset_sizes)
]

if __name__ == "__main__":
    ###########################
    # Init Vanilla CPU Intelex#
    ###########################
    # Run Vanilla Benchmark
    # vanilla_results = benchmark(offload="Vanilla", searchspace=generic_searchspace)

    ###################
    # Init GPU Intelex#
    ###################


    set_config(target_offload="gpu:0", working_memory=1024 * 15)
    print(get_config())
    # Run GPU Benchmark
    gpu_results = benchmark(
        offload="Intel ARC770", searchspace=supported_on_gpu_searchspace
    )

    ###################
    # Init CPU Intelex#
    ###################
    set_config(target_offload="cpu:0", working_memory=1024 * 100)
    print(get_config())
    # Run GPU Benchmark
    cpu_results = benchmark(
        offload="AMD 5950x SSE and AVX", searchspace=generic_searchspace
    )

    # Flatten list
    benchmark_results = gpu_results + cpu_results
    pd.DataFrame(benchmark_results).to_csv("results/intelex_results.csv", header=True)
    print(pd.DataFrame(benchmark_results))
