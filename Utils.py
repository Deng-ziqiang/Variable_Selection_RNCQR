import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from lifelines.utils import  concordance_index
import numpy as np
import pandas as pd
import math
import copy
import hashlib
import os

def get_hash(obj):
    return int(hashlib.sha256(str(obj).encode()).hexdigest(), 16) % (2**32 - 1)



from loss_function import custom_loss


def train_model(model, dataloader, optimizer, scheduler=None, num_epochs=25, device=None,h_n=1,tau=0.1,n_segments=10,lamba_=1,M=1):
    loss_instance = custom_loss(lambda_=lamba_, M=M, tau=tau, h_n=h_n, n_segments=n_segments)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')


    model.to(device)

    for epoch in range(num_epochs):



        model.train()

        running_loss = 0.0
        for batch in tqdm(dataloader):
            T = batch["T"].to(device)
            C = batch["C"].to(device)
            Y = batch["Y"].to(device)
            Delta = batch["censoring_indicator"].to(device)
            X = batch["X"].to(device)


            optimizer.zero_grad()

            f_X = model(X)

            total_loss = loss_instance.regularization_loss(model=model,X=X,Y=Y,Delta=Delta,f_X=f_X)
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * X.size(0)


        if scheduler:
            scheduler.step()

        epoch_loss = running_loss / len(dataloader.dataset)


        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())


    model.load_state_dict(best_model_wts)

    return model




def to_numpy(tensor):
    return tensor.detach().cpu().numpy()
class Metrics_RNCQR:
    def __init__(self,true_indices,dimension_x,selected_indices,tau,tol=1e-3):
        self.true_indices = true_indices
        self.total_indices = np.arange(dimension_x)
        self.tol = tol
        self.selected_indices = selected_indices
        self.tau = tau
    def metrics_result(self):
        TP, FP, TN, FN = self.compute_TP_FP_TN_FN(selected_indices=self.selected_indices)
        size = len(self.selected_indices)
        FPR, FNR = self.calculate_fpr_fnr(tp=TP, fp=FP, tn=TN, fn=FN)
        return size, FPR, FNR

    def compute_TP_FP_TN_FN(self, selected_indices):
        TP = len(set(selected_indices).intersection(set(self.true_indices)))
        FP = len(np.setdiff1d(selected_indices,
                              self.true_indices))
        TN = len(np.setdiff1d(np.setdiff1d(self.total_indices, selected_indices), self.true_indices))
        FN = len(np.setdiff1d(self.true_indices, selected_indices))
        return TP, FP, TN, FN
    def calculate_fpr_fnr(self,tp, fp, tn, fn):
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        return fpr, fnr
    def MMSE(self, y_true, y_pred, cen_indicator):







        y_true_log = y_true.flatten()
        y_pred_log = y_pred.flatten()
        temp = cen_indicator * (y_true_log - y_pred_log) ** 2
        MMSE = temp.sum() / cen_indicator.sum()
        return MMSE

    def UQL(self, y_true, y_pred, cen_indicator):







        y_true_log = y_true.flatten()
        y_pred_log = y_pred.flatten()

        obs_idx = cen_indicator.flatten() == 0
        if not obs_idx.any():
            UQL = 0
        else:
            UQL = ((y_true_log[obs_idx] - y_pred_log[obs_idx]) *
                   (self.tau - 1. * (y_pred_log[obs_idx] > y_true_log[obs_idx]))).mean()


        return UQL

    def MSE(self, y_true, y_pred):



        squared_diff = (y_true - y_pred) ** 2

        mse = np.mean(squared_diff)
        return mse

    def MSPE(self, y_true, y_pred):




        y_mean = np.mean(y_true)

        squared_diff = (y_pred - y_mean) ** 2

        mspe = np.mean(squared_diff)
        return mspe

class Metrics_evaluate:
    def __init__(self,true_indices,dimension_x,history_list,tau,device):
        self.true_indices = true_indices
        self.k = len(self.true_indices)
        self.history_list =history_list
        self.dimension_x = dimension_x
        self.tau = tau
        self.total_indices = np.arange(self.dimension_x)
        self.device = device
    def metrics_result(self):

        best_lambda, size, TP, FP, TN, FN, fpr, fnr = self.compute_metric_weighted_fpr()
        return best_lambda,size,fpr,fnr








    def compute_metric_weighted_fpr(self):
        results = []
        for history_item in self.history_list:
            selected_tensor = history_item.selected


            selected_indices = torch.where(selected_tensor)[0].tolist()
            size = len(selected_indices)
            lambda_selected = history_item.lambda_
            tp,fp,tn,fn = self.compute_TP_FP_TN_FN(selected_indices=selected_indices)
            fpr,fnr = self.calculate_fpr_fnr(tp=tp,fp=fp,tn=tn,fn=fn)
            weighted_fpr_fnr = self.weighted_sum_fpr_fnr(selected_indices=selected_indices,w1=1,w2=1)
            f1_score = self.f1_score(tp=tp,fp=fp,fn=fn)
            results.append({
                'selected_indices':selected_indices,
                'size':size,
                'lambda_selected':lambda_selected,
                'tp':tp,
                'fp':fp,
                'tn':tn,
                'fn':fn,
                'fpr':fpr,
                'fnr':fnr,
                'weighted_fpr_fnr':weighted_fpr_fnr,
                'f1_score':f1_score
            })


        best_results = self.find_best_by_F1_score(results=results)
        best_lambda = best_results['lambda_selected']
        best_size = best_results['size']
        tp_best = best_results['tp']
        fp_best = best_results['fp']
        tn_best = best_results['tn']
        fn_best = best_results['fn']
        FPR_best = best_results['fpr']
        FNR_best = best_results['fnr']
        print(f"{'*'*5} Best selected varaible = {best_results['selected_indices']},tp = {tp_best},fp = {fp_best},FPR = {FPR_best},FNR = {FNR_best},weighted sum of FPR_FNR = {best_results['weighted_fpr_fnr']} {'*'*5}")
        return best_lambda,best_size,tp_best,fp_best,tn_best,fn_best,FPR_best,FNR_best

    def find_best_by_F1_score(self,results):
        return  max(results,key=lambda x:x['f1_score'])
    def f1_score(self,tp,fp,fn):
        precision,recall = self.compute_precision_recall(tp=tp,fp=fp,fn=fn)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def compute_precision_recall(self,tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return precision, recall

    def save_selected_path(self,path_selected_variable):
        with open(path_selected_variable,'a') as f:
            for history_item in self.history_list:
                selected_tensor = history_item.selected
                selected_indices = torch.where(selected_tensor)[0].tolist()
                line_to_write = f"At lambda={history_item.lambda_}, selected variables are {selected_indices}\n"
                f.write(line_to_write)




    def compute_minsize_TP_FP(self):
        minsize = float('inf')
        best_TP = None
        best_FP = None
        best_TN = None
        best_FN = None
        best_lambda = None
        best_selected_indices = None
        OP = 0

        for history_item in self.history_list:
            selected_tensor = history_item.selected


            selected_indices = torch.where(selected_tensor)[0].tolist()

            if all(idx in selected_indices for idx in self.true_indices):
                current_size = len(selected_indices)
                if current_size < minsize:
                    minsize = current_size
                    best_TP = len(set(selected_indices).intersection(set(self.true_indices)))
                    best_FP = len(np.setdiff1d(selected_indices,
                                               self.true_indices))
                    best_TN = len(np.setdiff1d(np.setdiff1d(self.total_indices, selected_indices), self.true_indices))
                    best_FN = len(np.setdiff1d(self.true_indices, selected_indices))
                    best_lambda = history_item.lambda_
                    best_selected_indices = selected_indices
                if set(selected_indices) == set(self.true_indices):
                    OP = 1




        if minsize == float('inf'):
            print("No model includes all true variables.")
            return None
        else:
            print(
                f"Best Minsize is {minsize} with TP={best_TP}, FP={best_FP} , OP event = {OP} at lambda={best_lambda}, selected variables are {best_selected_indices}")
            return best_lambda,minsize, best_TP, best_FP,best_TN,best_FN,OP


    def compute_metrics_FPR(self):
        lowest_fpr_fnr = float('inf')
        best_size = None
        best_TP = None
        best_FP = None
        best_TN = None
        best_FN = None
        best_fpr = None
        best_fnr = None
        best_lambda = None
        best_selected_indices = None
        OP = 0

        for history_item in self.history_list:
            selected_tensor = history_item.selected


            selected_indices = torch.where(selected_tensor)[0].tolist()

            current_size = len(selected_indices)




            TP,FP,TN,FN = self.compute_TP_FP_TN_FN(selected_indices=selected_indices)
            lambda_selected = history_item.lambda_
            selected_indices_temp = selected_indices
            fpr,fnr = self.calculate_fpr_fnr(tp=TP,fp=FP,tn=TN,fn=FN)
            if fpr+fnr <lowest_fpr_fnr:
                best_size = current_size
                best_TP = TP
                best_TN = TN
                best_FN =FN
                best_FP = FP
                best_fpr = fpr
                best_fnr = fnr
                best_lambda = lambda_selected
                best_selected_indices = selected_indices_temp
        print(f"the best size = {best_size},with TP= {best_TP}, FP = {best_FP}, FPR = {best_fpr}, FNR = {best_fnr},at lambda = {best_lambda}, selected varaibles are {best_selected_indices}")
        return best_lambda,best_size,best_TP,best_FP,best_TN,best_FN,best_fpr,best_fnr


    def compute_TP_FP_TN_FN(self,selected_indices):
        TP = len(set(selected_indices).intersection(set(self.true_indices)))
        FP = len(np.setdiff1d(selected_indices,
                              self.true_indices))
        TN = len(np.setdiff1d(np.setdiff1d(self.total_indices, selected_indices), self.true_indices))
        FN = len(np.setdiff1d(self.true_indices, selected_indices))
        return TP,FP,TN,FN
    def calculate_fpr_fnr(self,tp, fp, tn, fn):
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        return fpr, fnr

    def weighted_sum_fpr_fnr(self,selected_indices, w1, w2):
        tp, fp, tn, fn = self.compute_TP_FP_TN_FN(selected_indices=selected_indices)
        fpr, fnr = self.calculate_fpr_fnr(tp=tp,fp=fp,tn=tn,fn=fn)
        return w1 * fpr + w2 * fnr

    def find_best_by_weighted_sum(self,results):

        return min(results, key=lambda x: x['weighted_fpr_fnr'])








    def MMSE(self,y_true, y_pred, cen_indicator):
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)
        cen_indicator = cen_indicator.to(self.device)




        y_true_log = y_true.flatten()
        y_pred_log = y_pred.flatten()
        temp = cen_indicator * (y_true_log - y_pred_log) ** 2
        MMSE = temp.sum() / cen_indicator.sum()
        return MMSE
    def UQL(self,y_true,y_pred,cen_indicator):
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)
        cen_indicator = cen_indicator.to(self.device)




        y_true_log = y_true.flatten()
        y_pred_log = y_pred.flatten()

        obs_idx = cen_indicator.flatten() == 0
        if not obs_idx.any():
            UQL = 0
        else:
            UQL = ((y_true_log[obs_idx] - y_pred_log[obs_idx]) *
                   (self.tau - 1. * (y_pred_log[obs_idx] > y_true_log[obs_idx]))).mean()


        return UQL

    def MSE(self,y_true,y_pred):
        y_true = y_true.to(self.device).flatten()
        y_pred = y_pred.to(self.device).flatten()

        squared_diff = (y_true - y_pred) ** 2

        mse = torch.mean(squared_diff)
        return mse

    def MSPE(self, y_true, y_pred):

        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)

        y_mean = torch.mean(y_true)

        squared_diff = (y_pred - y_mean) ** 2

        mspe = torch.mean(squared_diff)
        return mspe



    def position_prob(self):
        positions_of_interest = []
        for index, history_item in enumerate(self.history_list):
            selected_indices = torch.where(history_item.selected)[0].tolist()

            if set(selected_indices) == set(self.true_indices):
                positions_of_interest.append(index)

        return positions_of_interest
    def check_events(self, positions_of_interest):
        event_occurrences = []
        for pos in positions_of_interest:
            history_item = self.history_list[pos]
            selected_indices = torch.where(history_item['selected'])[0].tolist()
            if set(selected_indices) == set(self.true_indices):
                event_occurrences.append(1)
            else:
                event_occurrences.append(0)
        return event_occurrences







    def TP(self):
        best_TP = 0
        for history_item in self.history_list:
            selected_indices = torch.where(history_item.selected)[0].tolist()
            TP = len(set(selected_indices).intersection(set(self.true_indices)))
            best_TP = max(best_TP, TP)
        return best_TP

    def FP(self):
        best_FP = 0
        for history_item in self.history_list:
            selected_indices = torch.where(history_item.selected)[0].tolist()

            FP = len(np.setdiff1d(selected_indices, self.true_indices))
            best_FP = min(best_FP, FP)
        return best_FP

class metric_RandomForest:
    def __init__(self,true_indices,dimension_x,important_variables,tau,tol=1e-3):
        self.true_indices = true_indices
        self.total_indices = np.arange(dimension_x)
        self.tol = tol
        self.important_variables = important_variables
        self.importance_scores = self.extract_importance_scores()
        self.tau = tau
    def metrics_result(self):
        TP = self.TP()
        FP = self.FP()
        TN = self.TN()
        FN = self.FN()
        size = self.minisize()

        FPR = FP / max(FP + TN, 1)
        FNR = FN / max(FN + TP, 1)
        return size,FPR,FNR

    def TP(self):

        selected_indices = np.where(self.importance_scores > self.tol)[0]


        tp_count = len(np.intersect1d(selected_indices, self.true_indices))
        return tp_count
    def FP(self):

        selected_indices = np.where(self.importance_scores > self.tol)[0]

        fp_count = len(np.setdiff1d(selected_indices, self.true_indices))
        return fp_count
    def TN(self):

        selected_indices = np.where(self.importance_scores > self.tol)[0]
        TN = len(np.setdiff1d(np.setdiff1d(self.total_indices,selected_indices),self.true_indices))
        return TN
    def FN(self):

        selected_indices = np.where(self.importance_scores > self.tol)[0]
        FN = len(np.setdiff1d(self.true_indices, selected_indices))
        return FN

    def minisize(self):
        minisize = float('inf')




        selected_indices = np.where(self.importance_scores > self.tol)[0]
        print('selected indces')
        print(selected_indices)
        minisize = len(selected_indices)






        return minisize

    def prob_k_all(self):

        selected_indices = np.where(self.importance_scores > self.tol)[0]

        if set(selected_indices) == set(self.true_indices):
            return 1
        else:
            return 0

    def extract_importance_scores(self):
        return self.important_variables['importances_mean'].values

    def MMSE(self, y_true, y_pred, cen_indicator):







        y_true_log = y_true.flatten()
        y_pred_log = y_pred.flatten()


        temp = cen_indicator * np.abs(y_true_log - y_pred_log)
        MMSE = temp.sum() / cen_indicator.sum()
        return MMSE

    def UQL(self, y_true, y_pred, cen_indicator):







        y_true_log = y_true.flatten()
        y_pred_log = y_pred.flatten()
        obs_idx = cen_indicator.flatten() == 0

        if not obs_idx.any():
            UQL = 0
        else:
            UQL = ((y_true_log[obs_idx] - y_pred_log[obs_idx]) *
                   (self.tau - 1. * (y_pred_log[obs_idx] > y_true_log[obs_idx]))).mean()
        return UQL


    def MSE(self, y_true, y_pred):

        squared_diff = (y_true - y_pred) ** 2

        mse = np.mean(squared_diff)
        return mse

    def MSPE(self, y_true, y_pred):


        y_mean = np.mean(y_true)

        squared_diff = (y_pred - y_mean) ** 2

        mspe = np.mean(squared_diff)
        return mspe



def find_median_survival_time(survival_function):
    survival_times = survival_function.x
    survival_probabilities = survival_function.y


    median_index = np.where(survival_probabilities <= 0.5)[0]
    if median_index.size > 0:
        median_index = median_index[0]
        return survival_times[median_index]
    else:

        return survival_times[-1]


class metric_LinearCensored:
    def __init__(self,true_indices,dimension_x,variable_importance_score,tau,device,tol=1e-3):
        self.true_indices = true_indices
        self.total_indices = np.arange(dimension_x)
        self.tol = tol
        self.importance_scores = variable_importance_score
        self.tau = tau
        self.device = device

    def metrics_result(self):
        TP = self.TP()
        FP = self.FP()
        TN = self.TN()
        FN = self.FN()
        size = self.minisize()

        FPR = FP / max(FP + TN, 1)
        FNR = FN / max(FN + TP, 1)
        return size, FPR, FNR

    def TP(self):

        selected_indices = np.where(self.importance_scores > self.tol)[0]


        tp_count = len(np.intersect1d(selected_indices, self.true_indices))
        return tp_count
    def FP(self):

        selected_indices = np.where(self.importance_scores > self.tol)[0]

        fp_count = len(np.setdiff1d(selected_indices, self.true_indices))
        return fp_count

    def TN(self):

        selected_indices = np.where(self.importance_scores > self.tol)[0]
        TN = len(np.setdiff1d(np.setdiff1d(self.total_indices,selected_indices),self.true_indices))
        return TN
    def FN(self):

        selected_indices = np.where(self.importance_scores > self.tol)[0]
        FN = len(np.setdiff1d(self.true_indices, selected_indices))
        return FN

    def minisize(self):
        minisize = float('inf')


        selected_indices = np.where(self.importance_scores > self.tol)[0]

        minisize = len(selected_indices)






        return minisize
    def prob_k_all(self):

        selected_indices = np.where(self.importance_scores > self.tol)[0]

        if set(selected_indices) == set(self.true_indices):
            return 1
        else:
            return 0

    def MMSE(self, y_true, y_pred, cen_indicator):







        y_true_log = y_true.flatten()
        y_pred_log = y_pred.flatten()
        temp = cen_indicator * (y_true_log - y_pred_log) ** 2
        MMSE = temp.sum() / cen_indicator.sum()
        return MMSE

    def UQL(self, y_true, y_pred, cen_indicator):







        y_true_log = y_true.flatten()
        y_pred_log = y_pred.flatten()

        obs_idx = cen_indicator.flatten() == 0
        if not obs_idx.any():
            UQL = 0
        else:
            UQL = ((y_true_log[obs_idx] - y_pred_log[obs_idx]) *
                   (self.tau - 1. * (y_pred_log[obs_idx] > y_true_log[obs_idx]))).mean()


        return UQL

    def MSE(self, y_true, y_pred):



        squared_diff = (y_true - y_pred) ** 2

        mse = np.mean(squared_diff)
        return mse

    def MSPE(self, y_true, y_pred):




        y_mean = np.mean(y_true)

        squared_diff = (y_pred - y_mean) ** 2

        mspe = np.mean(squared_diff)
        return mspe




def clear_all_rows(conn):
    c = conn.cursor()
    c.execute("DELETE FROM results")
    conn.commit()



from rpy2.robjects import pandas2ri
pandas2ri.activate()



def generate_file_path(sample_size, dimension_x, type_indicator, censoring_rate, seed, base_directory,file_extension="csv"):

    file_name = f"size{sample_size}_dim{dimension_x}_type{type_indicator}_censor{censoring_rate}_seed{seed}.{file_extension}"

    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    file_path = os.path.join(base_directory, file_name)
    return file_path
def save_train_data(Y_train, censoring_indicator_train, X_train, file_path):

    Y_train_df = pd.DataFrame(Y_train, columns=['Y_train'])
    censoring_indicator_train_df = pd.DataFrame(censoring_indicator_train, columns=['censoring_indicator_train'])
    X_train_df = pd.DataFrame(X_train, columns=[f'X{i+1}' for i in range(X_train.shape[1])])

    train_data = pd.concat([Y_train_df, censoring_indicator_train_df, X_train_df], axis=1)

    train_data.to_csv(file_path, index=False)
def to_raw_string(path):

    return path.replace('\\', '\\\\')

def update_r_code(path_r_script, new_n, new_p,file_path_train):

    with open(path_r_script, "r", encoding="utf-8") as r_script:
        r_code = r_script.read()


    r_code = r_code.replace("n=40", f"n={new_n}")
    r_code = r_code.replace("p=50", f"p={new_p}")
    r_code = r_code.replace('train_file_path="train_data_temp.csv"', f'train_file_path="{file_path_train}"')

    return r_code

def extract_column(matrix, col_index):

    return matrix[:, col_index]


def remove_outliers(df, column):
    median = df[column].median()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    threoshold_value = 1
    lower_bound = Q1 - threoshold_value * IQR
    upper_bound = Q3 + threoshold_value * IQR
    df[column] = df[column].apply(lambda x: median if x < lower_bound or x > upper_bound else x)
    return df

