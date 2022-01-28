import argparse
from training.preprocess import preprocess
from experiment import run_random, run_classic_clean, run_boost_clean, run_cp_clean
from training.knn import KNN
import utils
import os

if __name__ == '__main__':
    val_size = 1000
    n_jobs = os.cpu_count()
    result_dir = "reproduce_result"
    space_dir = "data/data-reproduce"
    datasets = [d for d in os.listdir(space_dir) if d[0]!="."]

    for dataset in datasets:
        for mv_type in ["random", "system", "real"]:
            # load cached space
            cache_dir = os.path.join(space_dir, dataset, mv_type)

            if not os.path.exists(cache_dir):
                continue

            # model
            model = {
                "fn": KNN,
                "params": {"n_neighbors":3}
            }

            print("- Running", dataset, "with MV type", mv_type)
            data, info = utils.load_space(cache_dir)

            # preprocess data
            print("    - Preprocess data")
            data = preprocess(data)

            # vary val size
            data["X_val"] = data["X_val"][:val_size]
            data["y_val"] = data["y_val"][:val_size]
            info["val_size"] = len(data["X_val"])

            # directory to save the result
            save_dir = utils.makedir([result_dir, dataset, mv_type])

            # run baseline cleaning methods
            print("    - Run baseline methods")
            result_classic = run_classic_clean(data, model)
            result_bc = run_boost_clean(data, model)

            # save baseline results
            save_path = os.path.join(save_dir, "baseline.csv")
            dataset_info = {"dataset":info["dataset"], "mv_type":info["mv_type"]}
            utils.dicts_to_csv([dataset_info, result_classic, result_bc], save_path)

            # run cpclean
            print("    - Run CPClean")
            result_cp = run_cp_clean(data, model, n_jobs=n_jobs, debug_dir=save_dir)

            # run random clean
            print("    - Run RandomClean")
            result_random = run_random(data, model, n_jobs=n_jobs, debug_dir=save_dir)