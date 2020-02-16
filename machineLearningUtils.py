# coding: utf-8
import os, copy, functools
import numpy  as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics         import accuracy_score, f1_score, roc_auc_score, mean_squared_error

# スタッキング
class stacking():
    
    def _root_mean_squared_error(self, x, y):
        return np.sqrt(mean_squared_error(x, y))
    METRICS = {"acc"  : accuracy_score, "f1"   : f1_score,
               "auc"  : roc_auc_score,  "mse"  : mean_squared_error,
               "rmse" : _root_mean_squared_error}
    
    def __init__(self, cv, metric, seed=15):
        self.cv     = cv
        self.metric = self.METRICS[metric]
        self.seed   = seed
        self.stack_train  = []
        self.stack_test   = []
        self.stack_result = []
        if metric in ["f1","acc"]:
            self.metric_proba = False
        else:
            self.metric_proba = True
        
    def _append(self, appended, appending):
        return appended.append(appending, ignore_index=True)
    
    def _stack(self, train, test, result):
        self.stack_train.append(train)
        self.stack_test.append(test)
        self.stack_result.append(result)
    
    def predict(self, model, x):
        predicted = model.predict(x)
        if -1 < self.name.find("Classifier"):
            predicted_probability = 1 - model.predict_proba(x)[:,0]
        else:
            predicted_probability = predicted
        if self.metric_proba:
            predicted             = predicted_probability
        return predicted, predicted_probability
    
    def fit(self, model, x, t, test_x):
        df_stacked_predict = pd.DataFrame()
        df_stacked_result  = pd.DataFrame()
        start_time    = datetime.now()
        if hasattr(model, "__name__"):
            self.name = model.__name__
        else:
            self.name = model.__class__.__name__
        model_name    = str(len(self.stack_train) + 1) + "_" + self.name
        # train
        skf = StratifiedKFold(n_splits     = self.cv,
                              random_state = self.seed)
        for k, (train_idx, valid_idx) in enumerate(skf.split(x, t)):
            train_x = x.iloc[train_idx,:]
            train_t = t[train_idx]
            valid_x = x.iloc[valid_idx,:]
            valid_t = t[valid_idx]
            model.fit(train_x, train_t)
            train_metric, _           = self.predict(model, train_x)
            valid_metric, valid_stack = self.predict(model, valid_x)
            df_result  = pd.DataFrame({"k"                 : [k+1],
                                       "train_"+model_name : self.metric(train_t, train_metric),
                                       "valid_"+model_name : self.metric(valid_t, valid_metric)})
            df_predict = pd.DataFrame({"idx"      : valid_idx,
                                       model_name : valid_stack,
                                       "t"        : valid_t})
            df_stacked_result  = self._append(df_stacked_result,  df_result)
            df_stacked_predict = self._append(df_stacked_predict, df_predict)
        # test
        model.fit(x, t)
        train_metric, _ = self.predict(model, x)
        _, test_stack   = self.predict(model, test_x)
        df_result  = pd.DataFrame({"k"                 : ["all"],
                                   "train_"+model_name : self.metric(t, train_metric)})
        df_predict = pd.DataFrame({"idx"      : test_x.index,
                                   model_name : test_stack})
        df_stacked_result = self._append(df_stacked_result, df_result)
        self._stack(df_stacked_predict, df_predict, df_stacked_result)
        print("%s training end. time:%s" % (self.name, datetime.now()-start_time))
        
    def conversion_df(self):
        # each model result
        dfs = copy.deepcopy(self.stack_result)
        for idx, df in enumerate(dfs):
            if idx==0: continue
            dfs[idx] = df.drop("k", axis=1)
        df_stacked_result = pd.concat(dfs, axis=1)
        # each model predict
        df_stacked_train = functools.reduce(lambda x, y: pd.merge(x, y, on =["idx","t"]), self.stack_train)
        df_stacked_test  = functools.reduce(lambda x, y: pd.merge(x, y, on ="idx"),       self.stack_test)
        return df_stacked_result, df_stacked_train, df_stacked_test
    
    def fit_meta_model(self, model, train, test):
        train_x = train.drop(["idx","t"], axis=1)
        train_t = train.t
        test_x  = test.drop("idx", axis=1)
        # create meta model
        model.fit(train_x, train_t)
        train_metric, _ = self.predict(model, train_x)
        _, test_result  = self.predict(model, test_x)
        print("score : %s" % self.metric(train_t, train_metric))
        return test_result

