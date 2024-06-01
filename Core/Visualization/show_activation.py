import torch
from Common.Core.Visualization.graph_plot import plot
import torch.nn.functional as F
from d2l import torch as d2l


# 可视化激活函数函数
def compute_error_from_linear(x, y):
    return torch.sqrt(F.mse_loss(x, y))


def plot_activation_function(model, sequence, deep, device):
    x_w = torch.linspace(-500, 500, 500, device=device)
    x_n = torch.linspace(-10, 10, 100, device=device)
    y_w = model(x_w).detach().cpu()
    y_n = model(x_n).detach().cpu()
    x_w = x_w.cpu()
    x_n = x_n.cpu()
    error_w = compute_error_from_linear(x_w, y_w)
    error_n = compute_error_from_linear(x_n, y_n)
    params = [str(round(i.detach().float().item(), 3)) for i in model.parameters()]
    plot(x_w, y_w, xlabel="input", ylabel="output",
         title=f"deep:{deep}, layer:{sequence}, error:{round(error_w.item(), 3)},\n p:{', '.join(params)}")
    plot(x_n, y_n, xlabel="input", ylabel="output",
         title=f"deep:{deep}, layer:{sequence}, error:{round(error_n.item(), 3)},\n p:{', '.join(params)}")


def show_activation(model, activation, deep=0, device=d2l.try_gpu()):
    i = 1
    for layer in model:
        if isinstance(layer, activation):
            plot_activation_function(layer, i, deep, device=device)
            i += 1
        else:
            try:
                show_activation(layer, activation, deep + 1)
            except Exception as e:
                pass


def plot_activation_function_v2(model, sequence, deep, device, range_wide, range_narrow):
    x_w = torch.linspace(range_wide[0], range_wide[1], 500, device=device)
    x_n = torch.linspace(range_narrow[0], range_narrow[1], 100, device=device)
    y_w = model(x_w).detach().cpu()
    y_n = model(x_n).detach().cpu()
    x_w = x_w.cpu()
    x_n = x_n.cpu()
    error_w = compute_error_from_linear(x_w, y_w)
    error_n = compute_error_from_linear(x_n, y_n)
    params = [str(round(i.detach().float().item(), 3)) for i in model.parameters()]
    plot(x_w, y_w, xlabel="input", ylabel="output",
         title=f"deep:{deep}, layer:{sequence}, error:{round(error_w.item(), 3)},\n p:{', '.join(params)}")
    plot(x_n, y_n, xlabel="input", ylabel="output",
         title=f"deep:{deep}, layer:{sequence}, error:{round(error_n.item(), 3)},\n p:{', '.join(params)}")


def plot_activation_function_v3(model, sequence, device, range_wide, range_narrow):
    x_w = torch.linspace(range_wide[0], range_wide[1], 500, device=device)
    x_n = torch.linspace(range_narrow[0], range_narrow[1], 100, device=device)
    y_w = model(x_w).detach().cpu()
    y_n = model(x_n).detach().cpu()
    x_w = x_w.cpu()
    x_n = x_n.cpu()
    error_w = compute_error_from_linear(x_w, y_w)
    error_n = compute_error_from_linear(x_n, y_n)
    params = [str(round(i.detach().float().item(), 3)) for i in model.parameters()]
    plot(x_w, y_w, xlabel="input", ylabel="output",
         title=f"layer:{sequence}, error:{round(error_w.item(), 3)},\n p:{', '.join(params)}")
    plot(x_n, y_n, xlabel="input", ylabel="output",
         title=f"layer:{sequence}, error:{round(error_n.item(), 3)},\n p:{', '.join(params)}")


def data_expand(x, features):
    x_unsqueezed = x.unsqueeze(0)
    x_unsqueezed = torch.t(x_unsqueezed)
    x_expanded = x_unsqueezed.expand(-1, features)
    return x_expanded


def plot_activation_function_v4(model, sequence, device, range_wide, range_narrow):
    features = model.features
    x_w = torch.linspace(range_wide[0], range_wide[1], 500, device=device)
    x_w = data_expand(x_w, features)
    x_n = torch.linspace(range_narrow[0], range_narrow[1], 100, device=device)
    x_n = data_expand(x_n, features)
    print(x_w.shape)
    print(x_n.shape)
    y_w = model(x_w).detach().cpu()
    y_n = model(x_n).detach().cpu()
    x_w = x_w.cpu()
    x_n = x_n.cpu()
    error_w = compute_error_from_linear(x_w, y_w)
    error_n = compute_error_from_linear(x_n, y_n)
    print(error_w)
    print(error_n)
    # params = [str(round(i.detach().float().item(), 3)) for i in model.parameters()]
    plot(x_w[:, 0], y_w, xlabel="input", ylabel="output",
         title=f"layer:{sequence}\n")  # p:{', '.join(params)}")
    plot(x_n[:, 0], y_n, xlabel="input", ylabel="output",
         title=f"layer:{sequence}\n")  # p:{', '.join(params)}")


def show_activation_v2(model, activation, range_wide=[-2, 2], range_narrow=[-1, 1], deep=0, device=d2l.try_gpu()):
    if range_wide is None:
        range_wide = [-2, 2]
    i = 1
    for layer in model:
        if isinstance(layer, activation):
            print(layer)
            plot_activation_function_v2(layer, i, deep, device=device, range_wide=range_wide, range_narrow=range_narrow)
            i += 1
        else:
            try:
                show_activation_v2(layer, activation, range_wide, range_narrow, deep + 1)
            except Exception as e:
                pass


def find_layers_v2(model, type_of_layer):
    layers = []
    for layer in model:
        if isinstance(layer, type_of_layer):
            layers.append(layer)
        else:
            try:
                layers.extend(find_layers_v2(layer, type_of_layer))
            except Exception as e:
                pass
    return layers


def show_activation_v3(model, activation, range_wide=[-2, 2], range_narrow=[-1, 1], deep=0, device=d2l.try_gpu()):
    model.to(device)
    d2l.use_svg_display()
    layers = find_layers_v2(model, activation)
    i = 1
    for layer in layers:
        plot_activation_function_v3(layer, i, device, range_wide, range_narrow)
        i = i + 1


def show_activation_v4(model, activation, range_wide=[-2, 2], range_narrow=[-1, 1], deep=0, device=d2l.try_gpu()):
    model.to(device)
    d2l.use_svg_display()
    layers = find_layers_v2(model, activation)
    i = 1
    for layer in layers:
        plot_activation_function_v4(layer, i, device, range_wide, range_narrow)
        i = i + 1
