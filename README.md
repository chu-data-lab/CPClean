# CPClean

The official code for our VLDB 2021
paper: [Nearest Neighbor Classifiers over Incomplete Information: From Certain Answers to Certain Predictions](http://vldb.org/pvldb/vol14/p255-karlas.pdf)
.

## Installation

Install the package using pip

```
pip install -r requirements.txt
```

## Usage

### 1. Reproduce Experiment Results

To reproduce the main experiment results in our paper (i.e. Table 3), run the following
command:

```
python3 reproduce.py
```

**Results**

The results of a dataset with a type of missing values can be found in the folder ./reproduce_result/dataset_name/mv_type. Each folder has a baseline.csv containing the results of baseline methods and a CPClean.csv containing the results of CPClean.

The baseline.csv contains the following columns:
- **dataset**: name of the dataset
- **mv_type**: type of missing values
- **val_acc_{method}**: the validation accuracy using a cleaning method
- **test_acc_{method}**: the test accuracy using a cleaning method


<a name="cpclean_csv"></a>The CPClean.csv contains the following columns:
- **n_iter**: number of iterations
- **selection**: the id of the example selected to clean at this iteration
- **time**: time for selection at this iteration
- **percent_cp**: the percentage of validation examples CP'ed so far
- **percent_clean**: the percentage of examples cleaned so far
- **val_acc_gt, val_acc_dc**: the validation accuracy of ground truth and default cleaning
- **test_acc_gt, test_acc_dc**: the test accuracy of ground truth and default cleaning
- **test_acc_cpclean, val_acc_cpclean**: the test/val accuracy of CPClean at the current iteration

 Note that CPClean iteratively select an example to clean until all validation example CP'ed. The cpclean.csv contains the accuracy for all iterations. 


### 2. Construct CPClean Space

To build the space for running CPClean algorithm on a dataset, run the
following command. It will (1) split the dataset into train/val/test sets (2)
inject missing values
(if the mv_type is "random" or "systematic")  into the training set (3) run
cleaning algorithms to generate candidate repairs.

```
python build_space.py --data_dir <data_dir> --dataset <dataset_name> --mv_type <mv_type> --save_dir <space_dir>  --mv_prob <mv_prov> --val_size <val_size> --seed <seed>
```

**Before you run the command:** The raw data need to be stored in ``data_dir``
/``dataset_name``. For synthetic datasets (i.e., datasets to be injected with
missing values), it requires to provide a data.csv file containing the original
data and an info.json file containing the information of the dataset. See
data/datasets/Puma for example. For real datasets (i.e., datasets with real
missing values), it requires to provide a clean.csv containing the clean data,
a dirty.csv containing the dirty data and an info.json file containing the
information of the dataset. See data/datasets/BabyProduct for example.

**Arguments**

- **data_dir**: directory of the raw data.
- **dataset_name**: name of the dataset.
- **mv_type**: type of missing values. This can be "random", "systematic" or "real"
  . If "random", missing values will be completely randomly injected in to the
  features of the training data; if "systematic", missing values are more
  likely to be injected into important features; if "real", missing values will
  not be injected.
- **space_dir**: the directory to save the space.
- **mv_prob**: the probability of a cell to be missing, default 0.2.
- **val_size**: size of validation set, default 1400.
- **seed**: random seed, default 1.

**Results**

The results can be found in ``space_dir``/``dataset_name``/``mv_type`` with the
following files:
- **X_full.csv, y_full.csv**: features and labels of the original full dataset
- **X_train_clean.csv**: features of the clean training set.
- **X_train_dirty.csv**: features of the dirty training set with injected/real
  missing values.
- **X_train_repairs**: this folder contains the repaired training set using
  different cleaning methods.
- **X_train_ground_truth.csv**: features of the "ground truth" training set. For
  each example, we pick the candidate repair that is closest to its clean
  version (the corresponding example in X_train_clean.csv) as its ground truth.
  This is the possible world closest to the unknown clean world in the space.
- **y_train.csv**: labels of training set
- **X_val.csv, y_val.csv**: features and labels of the validation set
- **X_test.csv, y_test.csv**: feature and labels of the test set
- **indicator.csv**: boolean values indicating whether a cell is missing or not in the training set
- **info.json**: information of the dataset

### 3. Run CPClean Algorithm

To run our CPClean algorithm on a new dataset, first construct the CPClean
space on the dataset and then run the following command:

```
python cpclean.py --space_dir <space_dir> --dataset <dataset_name> --mv_type <mv_type> --result_dir <result_dir> --val_size <val_size> --n_jobs <n_jobs>
```

**Arguments**

- **space_dir**: the directory of the space.
- **dataset_name**: name of the dataset.
- **mv_type**: type of missing values. This can be "random", "systematic" or "real".
- **val_size**: size of validation set used to run cpclean, default 1000.
- **result_dir**: the directory to save the result.
- **n_jobs**: number of CPU. -1 means using all CPU, default -1.

**Results**

The results can be found in ``result_dir``/``dataset_name``/``mv_type``/CPClean.csv. The explanation of this file can be found [here](#cpclean_csv).