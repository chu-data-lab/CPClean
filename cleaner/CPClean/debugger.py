import numpy as np
from copy import deepcopy
import pandas as pd
from .knn_evaluator import KNNEvaluator
import utils

class Debugger(object):
    """docstring for Debugger"""
    def __init__(self, data, model, debug_dir):
        self.data = deepcopy(data)
        self.K = model["params"]["n_neighbors"]
        self.debug_dir = debug_dir
        self.logging = []
        self.n_dirty = self.data["X_train_mv"].isnull().values.any(axis=1).sum()
        self.n_val = len(self.data["X_val"])

    def init_log(self, percent_cc):
        self.gt_val_acc, self.gt_test_acc = \
            KNNEvaluator(self.data["X_train_gt"], self.data["y_train"], 
                        self.data["X_val"], self.data["y_val"], 
                        self.data["X_test"], self.data["y_test"]).score()
        self.X_train_mean = deepcopy(self.data["X_train_repairs"]["mean"])
        self.selection = []

        self.logging = []
        self.dc_val_acc, self.dc_test_acc = \
            KNNEvaluator(self.X_train_mean, self.data["y_train"], 
                self.data["X_val"], self.data["y_val"], 
                self.data["X_test"], self.data["y_test"]).score()

        self.logging.append([0, self.n_val, None, None, 
                             percent_cc, 0, 
                             self.gt_val_acc, self.dc_val_acc, self.dc_val_acc,
                             self.gt_test_acc, self.dc_test_acc, self.dc_test_acc
                            ])
        self.columns = ["n_iter", "n_val", "selection", "time", 
                        "percent_cp", "percent_clean", 
                        "val_acc_gt", "val_acc_gt", "val_acc_cpclean",
                        "test_acc_gt", "test_acc_dc", "test_acc_cpclean"]
        self.save_log()

    def save_log(self):
        logging_save = pd.DataFrame(self.logging, columns=self.columns)
        logging_save.to_csv(utils.makedir([self.debug_dir], "CPClean.csv"), index=False)

    def log(self, n_iter, sel, sel_time, percent_cc):
        self.selection.append(sel)

        percent_clean = len(self.selection) / self.n_dirty
        self.X_train_mean[sel] = self.data["X_train_gt"][sel]

        cpclean_val_acc, cpclean_test_acc = KNNEvaluator(self.X_train_mean, self.data["y_train"], 
                                                self.data["X_val"], self.data["y_val"], 
                                                self.data["X_test"], self.data["y_test"]).score()

        self.logging.append([n_iter, self.n_val, sel, sel_time, 
                             percent_cc, percent_clean, 
                             self.gt_val_acc, self.dc_val_acc, cpclean_val_acc,
                             self.gt_test_acc, self.dc_test_acc, cpclean_test_acc])

        self.percent_clean = percent_clean
        self.save_log()



