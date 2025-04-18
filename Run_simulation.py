import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools
import time
import optuna
import pandas as pd
from datetime import datetime,timedelta
from lifelines.utils import concordance_index

from sklearn.model_selection import KFold, train_test_split

from Data import dataset_positive
from Hyperparameters import read_configuration
from Model import residual_train
from Utils import get_hash, Metrics_RNCQR,to_numpy
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
metric_result_path = 'metric_Our_method.csv'

parameters = read_configuration("parameters.json")
parameters_our = parameters["data1"]



def simulation(param_simulation, seed):
    sample_size = param_simulation['sample_size']
    dimension_x = param_simulation['dimension_x']
    type_indicator = param_simulation['type_indicator']
    censoring_rate = param_simulation['censoring_rate']
    tau_train = param_simulation['tau_train']
    Dataset_instance = dataset_positive(sample_size=sample_size, dimension_X=dimension_x,
                                        correlation_rho=parameters_our["correlation_rho"],
                                        type_indicator=type_indicator,
                                        censoring_rate=censoring_rate, train_ratio=parameters_our["train_ratio"],
                                        random_seed=seed)
    X_train_tensor, y_train_tensor, delta_train_tensor,\
        X_test_tensor, y_test_tensor, delta_test_tensor = Dataset_instance.tensor_data_set_v2()

    true_indics_variable = Dataset_instance.true_indics()
    initial_params = {
        'num_epochs': 50,
        'lr_net': 0.001,
        'lr_theta': 0.001,
        'lr_W': 0.001,
        'l1_lambda': 0.05
    }
    final_params = {
        'num_epochs': 350,
        'lr_net': 0.001,
        'lr_theta': 0.0001,
        'lr_W': 0.0001,
        'l1_lambda': 0.01
    }

    final_params_base = {
        'num_epochs': 150,
        'lr_net': 0.001,
        'lr_theta': 0.0001,
        'lr_W': 0.0001,
        'l1_lambda': 0.01
    }
    hidden_dims_cv = (8,16,4)
    hidden_dims_final = (8,16,4)

    def cross_validation(X_train_tensor, y_train_tensor, delta_train_tensor,
                         k_values, tau_train,
                         initial_params, final_params,
                         param_simulation,
                         n_splits=5, seed=42):

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        best_k = None
        best_c_index = -np.inf


        for k in k_values:
            c_index_scores = []



            for fold, (train_fold_idx, val_fold_idx) in enumerate(kf.split(X_train_tensor)):

                X_train_fold = X_train_tensor[train_fold_idx]
                y_train_fold = y_train_tensor[train_fold_idx]
                delta_train_fold = delta_train_tensor[train_fold_idx]

                X_val_fold = X_train_tensor[val_fold_idx]
                y_val_fold = y_train_tensor[val_fold_idx]
                delta_val_fold = delta_train_tensor[val_fold_idx]


                selected_features, RNCQR_model, final_loss = residual_train(
                    X_train=X_train_fold,
                    delta_train=delta_train_fold,
                    Y_train=y_train_fold,
                    k=k,
                    tau=tau_train,
                    h_n=parameters_our["h_n"],
                    n_segments=parameters_our["n_segments"],
                    initial_train_params=initial_params,
                    final_train_params=final_params,
                    M=parameters_our["M"],
                    hidden_dims=hidden_dims_cv
                )


                y_predict_val = RNCQR_model.predict(X_test=X_val_fold)
                c_index = concordance_index(
                    event_times=to_numpy(y_val_fold),
                    predicted_scores=to_numpy(y_predict_val),
                    event_observed=to_numpy(delta_val_fold)
                )
                c_index_scores.append(c_index)



            avg_c_index = np.mean(c_index_scores)



            if avg_c_index > best_c_index:
                best_c_index = avg_c_index
                best_k = k




        if best_k is not None:
            print(f"Best k value determined: {best_k} with average c_index: {best_c_index:.4f}")
        else:
            print("Could not determine a best k value. Check errors and c-index results above.")

        return best_k


    k_values = range(1, 5)
    best_k = cross_validation(
        X_train_tensor=X_train_tensor,
        y_train_tensor=y_train_tensor,
        delta_train_tensor=delta_train_tensor,
        k_values=k_values,
        tau_train=tau_train,
        initial_params=initial_params,
        final_params=final_params,
        param_simulation=parameters_our,
        n_splits=parameters_our["n_splits_cv"],
        seed=42
    )

    k = best_k

    optuna_n_trials = parameters_our["number_trials"]
    optuna_val_size = parameters_our["optuna_val_size"]
    optuna_split_seed = 123




    indices = np.arange(X_train_tensor.shape[0])
    train_opt_idx, val_opt_idx = train_test_split(
        indices, test_size=optuna_val_size, random_state=optuna_split_seed
    )

    X_train_opt = X_train_tensor[train_opt_idx]
    y_train_opt = y_train_tensor[train_opt_idx]
    delta_train_opt = delta_train_tensor[train_opt_idx]

    X_val_opt = X_train_tensor[val_opt_idx]
    y_val_opt = y_train_tensor[val_opt_idx]
    delta_val_opt = delta_train_tensor[val_opt_idx]


    def objective(trial, X_train_opt, y_train_opt, delta_train_opt, X_val_opt, y_val_opt, delta_val_opt,
                  best_k, tau_train, initial_params, final_params_base,
                  h_n, n_segments, M, hidden_dims):



        lr_theta_trial = trial.suggest_float('lr_theta', 1e-5, 1e-1, log=True)
        l1_lambda_trial = trial.suggest_float('l1_lambda', 1e-5, 1e-1, log=True)


        final_params_trial = final_params_base.copy()
        final_params_trial['lr_theta'] = lr_theta_trial
        final_params_trial['l1_lambda'] = l1_lambda_trial



        try:

            _, RNCQR_model_trial, _ = residual_train(
                X_train=X_train_opt,
                delta_train=delta_train_opt,
                Y_train=y_train_opt,
                k=best_k,
                tau=tau_train,
                h_n=h_n,
                n_segments=n_segments,
                initial_train_params=initial_params,
                final_train_params=final_params_trial,
                M=M,
                hidden_dims=hidden_dims

            )


            y_predict_val = RNCQR_model_trial.predict(X_test=X_val_opt)
            c_index_val = concordance_index(
                event_times=to_numpy(y_val_opt),
                predicted_scores=to_numpy(y_predict_val),
                event_observed=to_numpy(delta_val_opt)
            )


            if np.isnan(c_index_val):

                return 0.5


            return c_index_val

        except Exception as e:
            print(f"Trial {trial.number}: Failed with error: {e}")
            print(f"Trial {trial.number}: Returning -inf.")

            return -np.inf


    print(f"\n--- Starting Optuna Hyperparameter Optimization ({optuna_n_trials} trials) ---")
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(
            trial,
            X_train_opt, y_train_opt, delta_train_opt,
            X_val_opt, y_val_opt, delta_val_opt,
            best_k,
            tau_train,
            initial_params,
            final_params_base,
            h_n=parameters_our["h_n"],
            n_segments=parameters_our["n_segments"],
            M=parameters_our["M"],
            hidden_dims=hidden_dims_final
        ),
        n_trials=optuna_n_trials,


    )


    print("\n--- Optuna Optimization Complete ---")
    try:
        best_hparams = study.best_params
        best_val_score = study.best_value
        print(f"Best validation C-index: {best_val_score:.4f}")
        print("Best hyperparameters found:")
        for key, value in best_hparams.items():
            print(f"  {key}: {value}")


        final_params_optimal = final_params_base.copy()
        final_params_optimal.update(best_hparams)

    except optuna.exceptions.TrialPruned as e:
        print(f"Optuna study finished, but the best trial was pruned: {e}")

        print("Warning: Optuna could not find optimal parameters. Using base final parameters.")
        final_params_optimal = final_params_base.copy()
    except Exception as e:
        print(f"An error occurred retrieving Optuna results: {e}")
        print("Warning: Optuna optimization might have failed. Using base final parameters.")
        final_params_optimal = final_params_base.copy()


    print("\n--- Training Final Model with Best k and Tuned Hyperparameters ---")
    print(f"Using k = {best_k}")
    print(f"Using Initial Params: {initial_params}")
    print(f"Using Optimal Final Params: {final_params_optimal}")

    selected_features_final, RNCQR_model_final, final_loss_final = residual_train(
        X_train=X_train_tensor,
        delta_train=delta_train_tensor,
        Y_train=y_train_tensor,
        k=best_k,
        tau=tau_train,
        h_n=parameters_our["h_n"],
        n_segments=parameters_our["n_segments"],
        initial_train_params=initial_params,
        final_train_params=final_params_optimal,
        M=parameters_our["M"],
        hidden_dims=hidden_dims_final

    )

    print("\n--- Final Model Training Complete ---")
    print(f"Selected Features ({len(selected_features_final)}): {selected_features_final}")
    print(f"Final Loss: {final_loss_final}")


    instance_metrics = Metrics_RNCQR(true_indices=true_indics_variable,
                                     dimension_x=dimension_x,
                                     selected_indices=selected_features_final,
                                     tau=tau_train
                                     )
    size, FPR, FNR = instance_metrics.metrics_result()
    y_predict = RNCQR_model_final.predict(X_test=X_test_tensor)
    c_index = concordance_index(
        event_times=to_numpy(y_test_tensor),
        predicted_scores=to_numpy(y_predict),
        event_observed=to_numpy(delta_test_tensor)
    )

    MSE = 1
    MSPE = 1
    MSE_noncensored = 1
    UQL = 1
    return size,FPR,FNR,MSE,MSPE,MSE_noncensored,UQL,c_index




def replicate(parameters,number_replication=parameters_our["number_replication"]):

    base_seed = get_hash(tuple(sorted(parameters.items())))


    metrics = pd.DataFrame(columns=["Size","FPR","FNR","MSE","MSPE","MSE_noncensored","UQL","c_index"])


    start_time = time.time()
    for i in range(number_replication):

        iteration_start_time = time.time()
        print(
            f"{'-' * 45} in ** {i}th ** replication, total replication: {number_replication} {'-' * 45}")

        current_seed = base_seed + i
        size,FPR,FNR,MSE,MSPE,MSE_noncensored,UQL,c_index = simulation(param_simulation=parameters,seed=current_seed)
        print(f"{'=' * 5} size:{size}, FPR: {FPR},FNR: {FNR}, c_index = {c_index} {'=' * 5} ")

        new_row = pd.DataFrame({
            "Size":size,
            "FPR":FPR,
            "FNR":FNR,
            "MSE":MSE,
            "MSPE":MSPE,
            "MSE_noncensored":MSE_noncensored,
            "UQL":UQL,
            "c_index":c_index
        },index=[0])
        metrics = pd.concat([metrics,new_row],ignore_index=True)


        iteration_end_time = time.time()
        iteration_duration = iteration_end_time - iteration_start_time
        total_elapsed_time = iteration_end_time - start_time
        average_time_per_iteration = total_elapsed_time / (i + 1)
        estimated_remaining_time = average_time_per_iteration * (number_replication - i - 1)
        estimated_end_time = datetime.now() + timedelta(seconds=estimated_remaining_time)
        print(f"{'='*15} 时间计算 {'='*15}")
        print(f"本次{i + 1}/{number_replication}的重复，消耗时间 {iteration_duration:.2f} seconds.")
        remaining_time_hours ,remainder = divmod(estimated_remaining_time,3600)
        print(f"预计剩余时间: {estimated_remaining_time:.2f} seconds, 即 {remaining_time_hours} 小时 ")
        print(f"所有重复预计结束时间: {estimated_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 15} 时间计算 {'=' * 15}")


    mean_metrics = metrics.mean()
    std_metrics = metrics.std()

    return mean_metrics,std_metrics
def run_whole_simulations(do_tuning=False):



    keys_of_interest = ["sample_size", "dimension_x", "type_indicator", "censoring_rate","tau_train"]
    combinations = list(itertools.product(*[parameters_our[key] for key in keys_of_interest]))


    results = pd.DataFrame(columns=["sample_size", "dimension_x", "type_indicator", "censoring_rate","tau_train",
                                    "Size_average","Size_sd","FPR_average","FNR_average",
                                    "MSE_average","MSE_sd","MSPE_average","MSPE_sd",
                                    "MSE_noncensored_average", "MSE_noncensored_sd",
                                    "UQL_average","UQL_sd","c_index_average","c_index_sd"])


    for combo in combinations:
        print(f"{'*'*50} combination = {combo} {'*'*50}")
        start_time = datetime.now()
        mean_metrics,std_metrics = replicate(
            dict(zip(keys_of_interest, combo)))

        paramer_combo = dict(zip(keys_of_interest, combo))
        sample_size = paramer_combo["sample_size"]
        dimension_x = paramer_combo["dimension_x"]
        type_indicator = paramer_combo["type_indicator"]
        censoring_rate = paramer_combo["censoring_rate"]
        tau = paramer_combo["tau_train"]

        new_row = pd.DataFrame([{
            "sample_size":sample_size, "dimension_x":dimension_x, "type_indicator":type_indicator, "censoring_rate":censoring_rate, "tau_train":tau,
            "Size_average":mean_metrics["Size"], "Size_sd":std_metrics["Size"], "FPR_average":mean_metrics["FPR"], "FNR_average":mean_metrics["FNR"],
            "MSE_average":mean_metrics["MSE"], "MSE_sd":std_metrics["MSE"], "MSPE_average":mean_metrics["MSPE"], "MSPE_sd":std_metrics["MSPE"],
            "MSE_noncensored_average":mean_metrics["MSE_noncensored"], "MSE_noncensored_sd":std_metrics["MSE_noncensored"],
            "UQL_average":mean_metrics["UQL"], "UQL_sd":std_metrics["UQL"],
            "c_index_average": mean_metrics["c_index"], "c_index_sd": std_metrics["c_index"]
        }])



        results = pd.concat([results,new_row],ignore_index=True)

        end_time = datetime.now()
        time_taken = (end_time - start_time).seconds
        print(f'this combination : {combo}, take time :{time_taken} seconds')
        print(f'v' * 100)

    results.to_csv(metric_result_path,index = False)


if __name__ == "__main__":
    run_whole_simulations(do_tuning=True)
