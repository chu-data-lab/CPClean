import argparse
from training.preprocess import preprocess
from experiment import run_random, run_classic_clean, run_boost_clean, run_cp_clean
from training.knn import KNN
import utils
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None)
parser.add_argument('--space_dir', default='cpclean_space')
parser.add_argument('--mv_type', default='random')
parser.add_argument('--result_dir', default="result/")
parser.add_argument('--val_size', default=1000, type=int)
parser.add_argument('--n_jobs', default=-1, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    # model
    model = {
        "fn": KNN,
        "params": {"n_neighbors":3}
    }

    if args.n_jobs < 0:
        args.n_jobs = os.cpu_count()
    
    # load space data    
    print("- Running on", args.dataset, "with MV type", args.mv_type)
    data, info = utils.load_space(os.path.join(args.space_dir, args.dataset, args.mv_type))

    # preprocess data
    print("    - Preprocess data")
    data = preprocess(data)

    # vary val size
    data["X_val"] = data["X_val"][:args.val_size]
    data["y_val"] = data["y_val"][:args.val_size]
    info["val_size"] = len(data["X_val"])

    # directory to save the result
    result_dir = utils.makedir([args.result_dir, args.dataset, args.mv_type])

    # run cpclean
    print("    - Run CPClean")
    result_cp = run_cp_clean(data, model, n_jobs=args.n_jobs, debug_dir=result_dir)
