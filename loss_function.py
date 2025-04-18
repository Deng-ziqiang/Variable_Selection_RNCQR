import torch


class custom_loss():
    def __init__(self,tau,h_n,n_segments,lambda_=1,M=1):
        self.lambda_ = lambda_
        self.M = M
        self.tau = tau
        self.h_n = h_n
        self.n_segments = n_segments
    def quantile_loss(self,f_X,Y):
        a = f_X - Y
        indicator = (a < 0).float()
        loss = ((self.tau - indicator) * a).sum()
        return loss

    def gaussian_kernel(self,u):
        return torch.exp(-torch.pow(u, 2))
    def compute_B(self,X):

        diff_matrix = X[:, None, :] - X[None, :, :]


        scaled_diff_matrix = diff_matrix / self.h_n



        kernel_values = self.gaussian_kernel(scaled_diff_matrix.flatten(end_dim=-2)).view_as(scaled_diff_matrix)


        summed_kernel_values = kernel_values.sum(dim=2)


        denominator = summed_kernel_values.sum(dim=1)


        B = summed_kernel_values / denominator[:, None]
        return B
    def compute_G_C_scalar(self,c,X,Y,Delta,index):
        B = self.compute_B(X)


        c_scalar = c.item()
        mask = (Y <= c_scalar).float() * (1 - Delta).float()
        numerator = torch.sum(mask * B[index, :])
        denominator = torch.sum((1 - Delta).float() * B[index, :])
        G_c = numerator / denominator

        return G_c

    def trapezoidal_integration_scalar(self,func, a, b, *args):
        h = (b.item() - a) / self.n_segments
        x_values = torch.linspace(a, b.item(), self.n_segments + 1)


        y_values_list = [func(x, *args) for x in x_values]
        y_values = torch.stack(y_values_list)
        y_values.requires_grad = True

        integral_estimate = h * (0.5 * y_values[0] + 0.5 * y_values[-1] + y_values[1:-1].sum())
        return integral_estimate

    def integral_loss(self,X, Y, Delta, f_X):
        integrals = []
        for i in range(X.size(0)):
            integral_value = self.trapezoidal_integration_scalar(self.compute_G_C_scalar, 0, f_X[i], X, Y, Delta,i)
            integrals.append(integral_value)
        loss = (1 - self.tau) * torch.sum(torch.stack(integrals))
        return loss
    def adapted_quantile_loss_1(self,X,Y,Delta,f_X):
        temp1 = self.integral_loss(X=X, Y=Y, Delta=Delta, f_X=f_X)
        temp2 = self.quantile_loss(f_X=f_X, Y=Y)

        adapted_loss = temp2 - temp1
        return adapted_loss

    def regularization_loss(self,model,X,Y,Delta,f_X):


        l1_reg = torch.norm(model.linear.weight, p=1)

        W_0 = model.residual[0].weight
        theta = model.linear.weight

        weight_constraint = torch.max(torch.abs(W_0), dim=0).values - self.M * torch.abs(theta)
        weight_constraint_penalty = torch.clamp(weight_constraint, min=0).sum()

        adapted_quantile_loss = self.quantile_loss(f_X=f_X,Y=Y)+self.integral_loss(X=X,Y=Y,Delta=Delta,f_X=f_X)
        total_loss = adapted_quantile_loss + self.lambda_*l1_reg+weight_constraint_penalty
        return total_loss
    def adapted_quantile_loss(self,X,Y,Delta,f_X):
        adapted_loss = self.quantile_loss(f_X=f_X, Y=Y)
        return adapted_loss

