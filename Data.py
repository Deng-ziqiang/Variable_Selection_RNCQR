import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch




class CustomDataset(Dataset):
    def __init__(self, data_frame):

        self.X = torch.tensor(data_frame[[f'X_{i}' for i in range(data_frame.shape[1] - 4)]].values, dtype=torch.float32)
        self.T = torch.tensor(data_frame["T"].values, dtype=torch.float32)
        self.C = torch.tensor(data_frame["C"].values, dtype=torch.float32)
        self.Y = torch.tensor(data_frame["Y"].values, dtype=torch.float32)
        self.censoring_indicator = torch.tensor(data_frame["censoring_indicator"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        sample = {
            "X": self.X[idx],
            "T": self.T[idx],
            "C": self.C[idx],
            "Y": self.Y[idx],
            "censoring_indicator": self.censoring_indicator[idx]
        }
        return sample



class dataset_positive():
    def __init__(self,sample_size,dimension_X,correlation_rho,type_indicator,censoring_rate,train_ratio,random_seed=None):
        self.sample_size = sample_size
        self.dimension_X = dimension_X
        self.correlation_rho = correlation_rho
        self.type_indicator = type_indicator
        self.random_seed = random_seed
        if self.random_seed is not  None:
            np.random.seed(self.random_seed)
        self.train_ratio = train_ratio

        self.dict_type = {
            "2": self.T_formula_2

        }

        self.dict_relevant_indices = {
            "1":[0,1,2,3],
            "2":[0,1,2,3],
            "3":[0,1,2],
            "4":[0,1]
        }

        self.data = self.generate_positive_data()


        self.T = self.data[:, 0]
        self.X = self.data[:, 1:]

        self.censoring_rate = censoring_rate
        self.tol = 0.01
        self.max_iterations = 5000000
        self.a = None
        self.b = np.max(self.T)
        self.C0 = 1


        self.Y,self.censoring_indicator,self.censored_time = self.generate_censored_data()

    def tensor_data_set(self):

        train_data, test_data = self.generate_data()
        new_train_data, val_data = self.split_train_validation(train_data = train_data, validation_ratio=0.2)

        X_train_tensor, y_train_tensor, delta_train_tensor = self.data_to_tensor(new_train_data)
        X_val,y_val,delta_val = self.data_to_tensor(val_data)
        X_test_tensor, y_test_tensor, delta_test_tensor = self.data_to_tensor(test_data)
        return X_train_tensor, y_train_tensor, delta_train_tensor, X_val,y_val,delta_val, X_test_tensor, y_test_tensor, delta_test_tensor

    def tensor_data_set_v2(self):

        train_data, test_data = self.generate_data()
        X_train_tensor, y_train_tensor, delta_train_tensor = self.data_to_tensor(train_data)
        X_test_tensor, y_test_tensor, delta_test_tensor = self.data_to_tensor(test_data)
        return X_train_tensor, y_train_tensor, delta_train_tensor, X_test_tensor, y_test_tensor, delta_test_tensor

    def generate_data(self):
        X = self.X
        T = self.T
        C = self.censored_time
        Y = self.Y
        censoring_indicator = self.censoring_indicator

        data_dict = {
            "T": T.flatten(),
            "C": C.flatten(),
            "Y": Y.flatten(),
            "censoring_indicator": censoring_indicator.flatten()
        }

        for i in range(X.shape[1]):
            data_dict[f'X_{i}'] = X[:, i]

        data = pd.DataFrame(data_dict)


        train_data, test_data = train_test_split(data, train_size=self.train_ratio, random_state=42)
        return train_data, test_data

    def split_train_validation(self, train_data, validation_ratio=0.2):

        new_train_data, val_data = train_test_split(
            train_data,
            test_size=validation_ratio,
            random_state=42
        )
        return new_train_data, val_data


    def data_RandomForest(self):
        X = self.X
        Y = self.Y
        censoring_indicator = self.censoring_indicator

        data_dict = {
            "Y": Y.flatten(),
            "censoring_time": censoring_indicator.flatten()
        }

        for i in range(X.shape[1]):
            data_dict[f'X_{i}'] = X[:, i]


        data = pd.DataFrame(data_dict)


        Y_with_indicator = np.array(list(zip(censoring_indicator == 0, Y)),
                                    dtype=[('censoring_indicator', 'bool'), ('Y', 'float64')])

        X_columns = sorted([col for col in data.columns if col.startswith('X_')],
                           key=lambda x: int(x.split('_')[1]))
        X_extracted = data[X_columns].values

        X_train, X_test, y_train, y_test = train_test_split(X_extracted, Y_with_indicator,
                                                            test_size=self.train_ratio, random_state=42)
        column_names = [f'X_{i}' for i in range(X_extracted.shape[1])]
        return X_train, y_train, X_test, y_test, column_names


    def data_LinearCensored(self):
        train_data, test_data = self.generate_data()

        Y_train = train_data['Y'].values
        censoring_indicator_train = train_data['censoring_indicator'].values


        Y_test = test_data['Y'].values
        censoring_indicator_test = test_data['censoring_indicator'].values


        X_columns_train = [col for col in train_data.columns if col.startswith('X_')]
        X_train = train_data[X_columns_train].values


        X_columns_test = [col for col in test_data.columns if col.startswith('X_')]
        X_test = test_data[X_columns_test].values

        return X_train, Y_train, censoring_indicator_train, X_test, Y_test, censoring_indicator_test

    def data_LinearCensored_tensor(self):
        train_data, test_data = self.generate_data()

        Y_train = train_data['Y'].values
        censoring_indicator_train = train_data['censoring_indicator'].values


        Y_test = test_data['Y'].values
        censoring_indicator_test = test_data['censoring_indicator'].values


        X_columns_train = [col for col in train_data.columns if col.startswith('X_')]
        X_train = train_data[X_columns_train].values


        X_columns_test = [col for col in test_data.columns if col.startswith('X_')]
        X_test = test_data[X_columns_test].values


        Y_train = torch.from_numpy(Y_train).type(torch.float32)
        censoring_indicator_train = torch.from_numpy(censoring_indicator_train).type(torch.int32)
        X_train = torch.from_numpy(X_train).type(torch.float32)

        Y_test = torch.from_numpy(Y_test).type(torch.float32)
        censoring_indicator_test = torch.from_numpy(censoring_indicator_test).type(torch.int32)
        X_test = torch.from_numpy(X_test).type(torch.float32)


        def to_device(tensor):
            return tensor.to('cuda') if torch.cuda.is_available() else tensor


        Y_train = to_device(Y_train)
        censoring_indicator_train = to_device(censoring_indicator_train)
        X_train = to_device(X_train)
        Y_test = to_device(Y_test)
        censoring_indicator_test = to_device(censoring_indicator_test)
        X_test = to_device(X_test)

        return X_train, Y_train, censoring_indicator_train, X_test, Y_test, censoring_indicator_test

    def data_to_tensor(self, dataframe):

        X_cols = [f'X_{i}' for i in range(self.dimension_X)]
        X = dataframe[X_cols].values
        Y = dataframe[['Y']].values
        censoring_indicator = dataframe['censoring_indicator'].values


        X_tensor = torch.from_numpy(X).float()
        Y_tensor = torch.from_numpy(Y).float()
        censoring_indicator_tensor = torch.from_numpy(censoring_indicator).float()
        return X_tensor, Y_tensor, censoring_indicator_tensor

    def generate_dataloaders(self, batch_size=32):
        train_data, test_data = self.generate_data()


        scaler = StandardScaler()
        train_data[[f'X_{i}' for i in range(self.dimension_X)]] = scaler.fit_transform(
            train_data[[f'X_{i}' for i in range(self.dimension_X)]])
        test_data[[f'X_{i}' for i in range(self.dimension_X)]] = scaler.transform(
            test_data[[f'X_{i}' for i in range(self.dimension_X)]])


        train_dataset = CustomDataset(train_data)
        test_dataset = CustomDataset(test_data)


        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def true_indics(self):
        return self.dict_relevant_indices[self.type_indicator]
    def generate_positive_data(self):
        data = np.zeros((self.sample_size, self.dimension_X + 1))
        for i in range(self.sample_size):
            T = -1
            while T <= 0:
                X = np.random.uniform(-1, 1, self.dimension_X)
                relevant_indices = self.dict_relevant_indices[self.type_indicator]
                T = self.dict_type[self.type_indicator](X,
                                                        relevant_indices)
            data[i, 0] = T
            data[i, 1:] = X
        return data


    def T_formula_2(self,X, relevant_indices):

        phi_X = X[relevant_indices[0]] + X[relevant_indices[1]] + X[relevant_indices[2]]+ X[relevant_indices[3]]
        epsilon = np.random.randn()
        T = np.exp(phi_X + epsilon)
        return T

    def generate_censored_data(self):

        return self.search_b_censoring_rate()



    def true_relevant_indices(self):
        if self.type_indicator in self.dict_relevant_indices:
            return self.dict_relevant_indices[self.type_indicator]
        else:
            raise ValueError(f"Invalid type_indicator: {self.type_indicator}")


    def search_b_censoring_rate(self):
        a = 0
        b = self.b
        n = len(self.T)
        iteration = 0
        step_size = 0.1

        while iteration < self.max_iterations:

            C = np.random.uniform(a, b, n)


            Y = np.minimum(C, self.T)
            delta = (self.T <= C).astype(int)


            censored_count = n - np.sum(delta)
            actual_censoring_rate = censored_count / n


            if abs(actual_censoring_rate - self.censoring_rate) <= self.tol:
                self.b = b



                break


            if actual_censoring_rate < self.censoring_rate:
                b -= b * step_size
            else:
                b += b * step_size

            step_size *= 0.95
            iteration += 1
        else:
            print(
                f"Warning: Maximum iterations reached. Closest achieved censoring rate: {actual_censoring_rate:.3f}")
        return Y, delta, C



