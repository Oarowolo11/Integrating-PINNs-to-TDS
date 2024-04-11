import torch
import torch.nn as nn

class Normalization_strat(nn.Module):
    def __init__(self, tensor_range, lb_range):
        super(Normalization_strat, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tensor_range = tensor_range.to(device)
        self.lb_range     = lb_range.to(device)
        self.twotimes = torch.tensor([2]).to(device)
        self.substractone = torch.tensor([1]).to(device)
    def forward(self, x):
        return self.twotimes*(x-self.lb_range)/self.tensor_range-self.substractone
    
class Unormalization_strat(nn.Module):
    def __init__(self, tensor_range, lb_range):
        super(Unormalization_strat, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tensor_range = tensor_range.to(device)
        self.lb_range     = lb_range.to(device)
        self.twotimes = torch.tensor([2]).to(device)
        self.substractone = torch.tensor([1]).to(device)
    def forward(self, x):
        return (x+self.substractone)*self.tensor_range/self.twotimes+self.lb_range
    
class FCN(nn.Module):
    "Defines a fully-connected network in PyTorch"
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, norm_range, lb_range, range_unormalization, lb_range_unormalization) -> None:
        super().__init__()
        torch.set_default_dtype(torch.float64)
        activation = nn.Tanh
        self.norm = Normalization_strat(torch.tensor(norm_range, dtype=torch.float64), torch.tensor(lb_range, dtype=torch.float64))
        self.unorm = Unormalization_strat(torch.tensor(range_unormalization, dtype=torch.float64), torch.tensor(lb_range_unormalization, dtype=torch.float64))
        self.fcs = nn.Sequential(*[self.norm,
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        torch.manual_seed(123)
        nn.init.xavier_normal_(self.fcs[1].weight)
        for module in self.fch:
            if isinstance(module[0], nn.Linear):
                nn.init.xavier_normal_(module[0].weight)
        nn.init.xavier_normal_(self.fce.weight)
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        x = self.unorm(x)
        return x