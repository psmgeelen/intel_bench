import numpy as np
import pandas as pd
from glob2 import glob
import timeit
from tqdm import tqdm
from itertools import product
from cuml.cluster import DBSCAN, KMeans
from cuml.decomposition import PCA
from cuml.linear_model import LinearRegression
from cuml.neighbors import KNeighborsRegressor
from cuml.ensemble import RandomForestClassifier, RandomForestRegressor
from cuml.svm import SVC
import cudf

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
        try:
            time = timeit.timeit(
                stmt=f"{algorithm}(X=data[:{size},:])", globals=globals(), number=iterations
            )
        except:
            print(f"failed to run, not enough parameters {size} for {algorithm}")
            time = None

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

generic_searchspace = [item for item in product(algorithms, subset_sizes)]

if __name__ == "__main__":
    ###############
    # Init RAPIDS #
    ###############
    # Run Vanilla Benchmark
    rapids = benchmark(offload="Rapids", searchspace=generic_searchspace)

    pd.DataFrame(rapids).to_csv("results/rapids_results.csv", header=True)
    print(pd.DataFrame(rapids))
