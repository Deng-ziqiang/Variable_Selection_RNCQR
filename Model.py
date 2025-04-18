import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



from loss_function import custom_loss

class CustomNet(nn.Module):
    def __init__(self, p, hidden_dims=(16,), M=1.0):
        super(CustomNet, self).__init__()
        self.p = p
        self.M = M
        self.W = nn.Parameter(torch.ones(p))

        self.theta = nn.Parameter(torch.ones(p))

        layers = []
        in_dim = p
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        if X.shape[1] != self.p:

            pass


        if X.shape[1] == self.p:
            X_scaled = X * self.W
            skip = torch.matmul(X, self.theta.unsqueeze(-1))

        else:

            X_scaled = X * self.W
            skip = torch.matmul(X, self.theta.unsqueeze(-1))

        f_out = self.network(X_scaled)
        return f_out + skip

    def predict(self, X_test):
        self.eval()
        with torch.no_grad():
            preds = self.forward(X_test)
        return preds

    def prune_weights(self, threshold=0.001):
        with torch.no_grad():
            self.W.data = torch.where(torch.abs(self.W.data) < threshold,
                                      torch.tensor(0.0, device=self.W.device),
                                      self.W.data)


def project_parameters(model):
    with torch.no_grad():
        allowed = model.M * torch.abs(model.theta)

        current_p = model.W.shape[0]
        if allowed.shape[0] != current_p:

            print(f"Warning: Shape mismatch in project_parameters. W: {model.W.shape}, theta: {model.theta.shape}")

            pass

        clamped_W = torch.clamp(model.W.data, min=-allowed, max=allowed)
        model.W.data = clamped_W



def train_model_original(model, X_train, delta_train, Y_train,
                tau,
                h_n=0.3,
                n_segments=100,
                num_epochs=100,
                lr_net=0.001,
                lr_theta=0.0001,
                lr_W=0.0001,
                l1_lambda=1.0,
                verbose=True):
    device = X_train.device
    model.to(device)
    Y_train = Y_train.to(device)
    if delta_train is not None:
        delta_train = delta_train.to(device)

    loss_instance = custom_loss(tau=tau, h_n=h_n, n_segments=n_segments)
    optimizer = optim.Adam([
        {'params': model.network.parameters(), 'lr': lr_net},
        {'params': [model.theta], 'lr': lr_theta},
        {'params': [model.W], 'lr': lr_W}
    ])



    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()


        current_p = model.p
        if X_train.shape[1] != current_p:



            pass

        outputs = model(X_train)

        loss_main = loss_instance.adapted_quantile_loss(
            X=X_train,
            Y=Y_train,
            Delta=delta_train,
            f_X=outputs
        )


        scaling_factor = (epoch + 1) / num_epochs
        loss_l1 = scaling_factor * l1_lambda * torch.norm(model.theta, 1)

        loss = loss_main + loss_l1
        loss.backward()

        optimizer.step()
        project_parameters(model)

        penalty = scaling_factor * lr_theta * l1_lambda
        with torch.no_grad():
            theta_old = model.theta.data
            model.theta.data = torch.sign(theta_old) * torch.clamp(torch.abs(theta_old) - penalty, min=0.0)






    return model


def train_model(model, X_train, delta_train, Y_train,
                tau,
                h_n=0.3,
                n_segments=100,
                num_epochs=100,
                lr_net=0.001,
                lr_theta=0.0001,
                lr_W=0.0001,
                l1_lambda=1.0,
                verbose=True):
    device = X_train.device
    model.to(device)
    Y_train = Y_train.to(device)
    if delta_train is not None:
        delta_train = delta_train.to(device)

    loss_instance = custom_loss(tau=tau, h_n=h_n, n_segments=n_segments)
    optimizer = optim.Adam([
        {'params': model.network.parameters(), 'lr': lr_net},
        {'params': [model.theta], 'lr': lr_theta},
        {'params': [model.W], 'lr': lr_W}
    ])


    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        current_p = model.p
        if X_train.shape[1] != current_p:


            pass

        outputs = model(X_train)

        loss_main = loss_instance.adapted_quantile_loss(
            X=X_train,
            Y=Y_train,
            Delta=delta_train,
            f_X=outputs
        )

        scaling_factor = (epoch + 1) / num_epochs
        loss_l1 = scaling_factor * l1_lambda * torch.norm(model.theta, 1)

        loss = loss_main + loss_l1
        loss.backward()

        optimizer.step()
        project_parameters(model)

        penalty = scaling_factor * lr_theta * l1_lambda
        with torch.no_grad():
            theta_old = model.theta.data
            model.theta.data = torch.sign(theta_old) * torch.clamp(torch.abs(theta_old) - penalty, min=0.0)

    model.eval()
    with torch.no_grad():
        final_outputs = model(X_train)
        final_loss_main = loss_instance.adapted_quantile_loss(
            X=X_train, Y=Y_train, Delta=delta_train, f_X=final_outputs
        )
        final_loss_l1 = 1.0 * l1_lambda * torch.norm(model.theta, 1)
        final_loss = final_loss_main + final_loss_l1

    return model, final_loss.item()


def residual_compute(X_subset, Y_train, epochs=50, lr=0.01):
    k = X_subset.shape[1]
    if k == 0:
        return torch.zeros_like(Y_train)

    device = X_subset.device
    residual_result = nn.Linear(k, 1, bias=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(residual_result.parameters(), lr=lr)
    Y_train = Y_train.to(device)

    for epoch in range(epochs):
        residual_result.train()
        optimizer.zero_grad()
        outputs = residual_result(X_subset)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()

    residual_result.eval()
    with torch.no_grad():
        predictions = residual_result(X_subset)
    return predictions



def residual_train(X_train, delta_train, Y_train, k,
                                tau, h_n, n_segments,
                                initial_train_params,
                                final_train_params,
                                M=1.0, hidden_dims=(16,)):

    n_samples, p_features = X_train.shape
    device = X_train.device

    initial_model = CustomNet(p=p_features, hidden_dims=hidden_dims, M=M).to(device)
    initial_model,final_loss =     train_model(initial_model, X_train, delta_train, Y_train,
                tau=tau, h_n=h_n, n_segments=n_segments,
                verbose=True,
                **initial_train_params)

    with torch.no_grad():
        initial_i_residual = torch.abs(initial_model.theta.data).cpu().numpy()

    in_resi_r = np.argsort(initial_i_residual)[::-1].tolist()

    selected_indices = []
    remaining_indices = in_resi_r[:]


    t_fea = remaining_indices.pop(0)
    selected_indices.append(t_fea)

    current_Y_residual = Y_train.clone().to(device)

    for i in range(1, k):
        if not remaining_indices:
            break


        X_selected_subset = X_train[:, selected_indices].to(device)
        Y_pred_subset = residual_compute(X_selected_subset, Y_train)
        current_Y_residual = Y_train.to(device) - Y_pred_subset

        residual_co = []
        indices_to_evaluate = remaining_indices[:]

        for idx in indices_to_evaluate:
            indi_fea_re = X_train[:, idx].unsqueeze(1).to(device)



            feat_centered = indi_fea_re - torch.mean(indi_fea_re)
            res_centered = current_Y_residual - torch.mean(current_Y_residual)

            std_feat = torch.std(feat_centered)
            std_res = torch.std(res_centered)

            if std_feat > 1e-9 and std_res > 1e-9:
                corr = torch.sum(feat_centered * res_centered) / (std_feat * std_res * n_samples)
            else:
                corr = torch.tensor(0.0)

            residual_co.append(torch.abs(corr).item())

        if not residual_co:
            break


        in_can_b_l = np.argmax(residual_co)
        indi_fea_g = indices_to_evaluate[in_can_b_l]


        selected_indices.append(indi_fea_g)
        remaining_indices.remove(indi_fea_g)


    final_selected_indices = sorted(selected_indices)

    if not final_selected_indices:
        return [], None

    final_model = CustomNet(p=p_features, hidden_dims=hidden_dims, M=M).to(device)

    final_model.network.load_state_dict(initial_model.network.state_dict())

    selected_mask = torch.zeros(p_features, dtype=torch.bool, device=device)
    selected_mask[final_selected_indices] = True
    unselected_mask = ~selected_mask

    with torch.no_grad():
        final_model.W.data[unselected_mask] = 0.0
        final_model.theta.data[unselected_mask] = 0.0

        final_model.W.data[selected_mask] = 1.0
        final_model.theta.data[selected_mask] = 1.0



    return final_selected_indices, final_model,final_loss









































































