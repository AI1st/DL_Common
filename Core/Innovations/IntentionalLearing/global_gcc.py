import torch
from torch import nn
from d2l import torch as d2l


class Relpu_s(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b):
        # 前向传播逻辑
        mask = (x > 0).float()
        temp0 = 1 + mask * x
        temp1 = torch.pow(temp0, b - 1)
        temp2 = temp1 * temp0
        result = x + a * (temp2 - 1)
        ctx.save_for_backward(a, b, mask, temp0, temp1, temp2)  # 保存输入张量，以备反向传播使用
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播逻辑
        a, b, mask, temp0, temp1, temp2 = ctx.saved_tensors  # 获取保存的输入张量
        grad_x = grad_a = grad_b = None

        if ctx.needs_input_grad[0]:
            # grad_x = grad_output * (1 + mask * a * b * torch.pow(1 + x, b - 1))
            grad_x = grad_output * (1 + mask * temp1 * a * b)

        if ctx.needs_input_grad[1]:
            # grad_a = grad_output * torch.pow(1 + x, b) * mask - mask
            grad_a = grad_output * (temp2 - 1)

        if ctx.needs_input_grad[2]:
            # grad_b = grad_output * a * torch.pow(1 + x, b) * torch.log(1 + x * mask) * mask
            grad_b = grad_output * a * torch.log(temp0) * temp2

        return grad_x, grad_a, grad_b


def relpu(x, a, b):
    output = Relpu_s.apply(x, a, b)
    return output


def sigmoid_distribution(s, a, b, c):
    return 1 / (1 + torch.exp(-a * (s - b))) * (1 - c) + c


class RelpuGlob(nn.Module):
    # version RelpuGlob
    def __init__(self, features, u_a=0, u_b=1, sigma_a=0.05, sigma_b=0.05, dim=1):
        super().__init__()
        self.features = features
        self.weight_a = nn.Parameter(sigma_a * (torch.randn(features)) + u_a)
        self.weight_b = nn.Parameter(sigma_b * (torch.randn(features)) + u_b)
        self.dim = dim

    def forward(self, x):
        view = [self.features if i == self.dim else 1 for i in range(len(x.shape))]
        x = relpu(x, self.weight_a.view(view), self.weight_b.view(view))
        return x


class GCC(nn.Module):
    # version GCC
    def __init__(self, softness_ratio, rejection_rate, compromise_ratio,
                 type="noise", p_rate=1, base=0, disturbing_rate=0):
        """_summary_
        Args:
            softness (float): _description_
            rejection_rate (float): _description_
            type (str, optional): _description_. Defaults to "noise", optional to "dicision"
        """
        super().__init__()
        self.type = type
        self.softness_ratio = torch.tensor(softness_ratio)
        self.rejection_rate = torch.tensor(rejection_rate)
        self.compromise_ratio = torch.tensor(compromise_ratio)
        self.p_rate = p_rate
        self.base = base
        self.disturbing_rate = disturbing_rate

    def forward(self, x):
        if self.training:
            mask = torch.rand(*x.shape, device=x.device)
            mask = sigmoid_distribution(
                mask,
                self.softness_ratio,
                self.rejection_rate * self.p_rate + self.base,
                self.compromise_ratio)
            mask_mean = mask.mean()
            randn_bias = torch.randn(*x.shape, device=x.device) * x.mean() * self.disturbing_rate
            x = x * mask * 1 / mask_mean + randn_bias
        return x


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def init_Relpu_glob(u_a, u_b, sigma_a, sigma_b, a_type=RelpuGlob):
    def init(m):
        if type(m) == a_type:
            m.weight_a = nn.Parameter(u_a + sigma_a * torch.randn(m.features))
            m.weight_b = nn.Parameter(u_b + sigma_b * torch.randn(m.features))

    return init


################################TRAIN##############################


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


def rectified_gcc_universal(net, type,
                            softness_ratio='default', rejection_rate='default', compromise_ratio='default',
                            controlling_unit=GCC):
    layers = find_layers_v2(net, controlling_unit)  # 此处当名称变化时应该修改
    layers_in_type = []
    for layer in layers:
        if type == layer.type:
            layers_in_type.append(layer)
            if softness_ratio != 'default':
                layer.softness_ratio = torch.tensor(softness_ratio)
            if rejection_rate != 'default':
                layer.rejection_rate = torch.tensor(rejection_rate)
            if compromise_ratio != 'default':
                layer.compromise_ratio = torch.tensor(compromise_ratio)
    return layers_in_type


class PID:
    def __init__(self, Kp, Ki, Kd, clear_threshold=0.03, windup_guard=1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Pterm = 0
        self.Iterm = 0
        self.Dterm = 0
        self.clear_threshold = clear_threshold
        self.windup_guard = windup_guard
        self.last_error = 0

    def __call__(self, train_acc, test_acc):
        return self.pid(train_acc, test_acc)

    def pid(self, train_acc, test_acc):
        e = train_acc - test_acc
        delta_e = e - self.last_error
        self.last_error = e

        self.Pterm = e
        self.Iterm += e
        self.Dterm = delta_e

        if abs(e) < self.clear_threshold:
            self.clear_Iterm()  # 清除积分项（积分项减半）
        if self.Dterm > self.windup_guard:
            self.Dterm = self.windup_guard  # 防止积分项过大

        return self.Kp * self.Pterm + self.Ki * self.Iterm + self.Kd * self.Dterm

    def clear_Iterm(self):
        self.Iterm = self.Iterm / 2


def rt_for_activation_function_v7(model, activation_version, a_control, b_control1, b_control2, b_control3):
    layers = find_layers_v2(model, activation_version)
    a_sum = 0
    b_sum = 0
    i = 0
    j = 0
    for layer in layers:
        for name, param in layer.named_parameters():
            if name == "weight_a":
                a_sum += a_control * torch.sum(torch.square(param))
                i += layer.features
            elif name == "weight_b":
                b_sum += b_control1 * torch.sum(torch.pow(b_control2, b_control3 * torch.square(param - 1)))
                j += layer.features
    if i == 0:
        return 0
    return a_sum / i + b_sum / j


def rt_for_activation_function_difference(model, activation_version, k1, k2):
    layers = find_layers_v2(model, activation_version)
    a_sum = 0
    b_sum = 0
    for layer in layers:
        for name, param in layer.named_parameters():
            if name == "weight_a":
                a_sum += k1 * torch.var(param)
            elif name == "weight_b":
                b_sum += k2 * torch.var(param)
    return a_sum + b_sum


def write_pid_control(control_result, logfile_name="pid_data.logs"):
    with open(logfile_name, "a", encoding="utf-8") as f:
        f.write(str(control_result))
        f.write("\n")


def train_relpu_gcc_gpus(net, train_iter, test_iter, num_epochs, lr, num_gpus, activation, controlling_unit,
                         a_control, b_control1, b_control2, b_control3, weight_decay=0,
                         Kp=0.5, Ki=0.5, Kd=0.5, clear_threshold=0.03, windup_guard=1,
                         softness_ratio_init=30, decision_rejection_rates_init=0.5, compromise_ratio_init=0.1,
                         path="", model_name="", save=False, logfile_name="pid_data.logs"):
    """

    :param net:
    :param train_iter:
    :param test_iter:
    :param num_epochs:
    :param lr:
    :param num_gpus:
    :param a_control:
    :param b_control1:
    :param b_control2:
    :param b_control3:
    :param weight_decay:
    :param path:
    :param model_name:
    :param save:
    :return:
    """
    train_l = None
    train_acc = None
    test_acc = None
    metric = None

    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    print('training on', *devices)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    ###########################PID试验区###########################
    rectified_gcc_universal(
        net.module, "dicision",
        softness_ratio=softness_ratio_init,
        rejection_rate=decision_rejection_rates_init,
        compromise_ratio=compromise_ratio_init,
        controlling_unit=controlling_unit)
    pid = PID(Kp, Ki, Kd, clear_threshold, windup_guard)
    print("PID initialized!", f"Kp:{Kp}, Ki:{Ki}, Kd:{Kd}")
    print(f"softness_ratio_init:{softness_ratio_init}")
    print(f"decision_rejection_rates_init:{decision_rejection_rates_init}")
    print(f"compromise_rate_init:{compromise_ratio_init}")
    write_pid_control("#################start################", logfile_name)
    write_pid_control(f"model name:{model_name}", logfile_name)
    write_pid_control(f"lr:{lr}, a_control:{a_control}, b_control:{b_control1}", logfile_name)
    write_pid_control(f"Kp:{Kp}, Ki:{Ki}, Kd:{Kd}", logfile_name)
    write_pid_control(f"clear_threshold:{clear_threshold}, windup_guard:{windup_guard}", logfile_name)
    write_pid_control(
        f"softness_ratio_init:{softness_ratio_init}, decision_rejection_rates_init:{decision_rejection_rates_init}",
        logfile_name)
    write_pid_control(f"compromise_ratio_init:{compromise_ratio_init}", logfile_name)
    ###########################PID试验区###########################
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            y_hat = net(X)
            #########################################################
            # 增加了对激活函数的正则项
            rt = rt_for_activation_function_v7(net.module, activation,
                                               a_control,
                                               b_control1, b_control2, b_control3)
            loss_only = loss(y_hat, y)
            l = loss_only + rt
            #########################################################
            l.backward()
            optimizer.step()

            with torch.no_grad():
                # 仅仅使用无rt的loss，避免其影响最终的误差衡量
                metric.add(loss_only * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        ###########################PID试验区###########################
        rejection_rate = pid(train_acc, test_acc) + decision_rejection_rates_init  # 注意；train_acc和test_acc均为浮点数
        rectified_gcc_universal(net.module, "dicision",
                                rejection_rate=rejection_rate, controlling_unit=controlling_unit)
        write_pid_control(f"test:{test_acc}, train:{train_acc}", logfile_name)
        write_pid_control(f"train_loss:{train_l}", logfile_name)
        write_pid_control(">>" + str(rejection_rate), logfile_name)
        ###########################PID试验区###########################
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on ', *devices)

    if save is True:
        train_acc_s = train_acc * 1000
        test_acc_s = test_acc * 1000
        torch.save(
            net.state_dict(),
            path + f"/{model_name}_tr{train_acc_s:.0f}_te{test_acc_s:.0f}"
                   f"_lr{lr}_rt{a_control}_{b_control1}_{b_control2}_wd{weight_decay}.params")

    ###########################PID试验区###########################
    write_pid_control("total time:" + str(timer.sum()), logfile_name)
    write_pid_control("#################end################", logfile_name)
    ###########################PID试验区###########################
    return rejection_rate


def train_relpu_gcc_gpus_v2(net, train_iter, test_iter, num_epochs, lr, num_gpus, activation, controlling_unit,
                            a_control, b_control1, b_control2, b_control3, k1, k2, weight_decay=0,
                            Kp=0.5, Ki=0.5, Kd=0.5, clear_threshold=0.03, windup_guard=1,
                            softness_ratio_init=30, decision_rejection_rates_init=0.5, compromise_ratio_init=0.1,
                            path="", model_name="", save=False, logfile_name="pid_data.logs"):
    """

    :param net:
    :param train_iter:
    :param test_iter:
    :param num_epochs:
    :param lr:
    :param num_gpus:
    :param a_control:
    :param b_control1:
    :param b_control2:
    :param b_control3:
    :param weight_decay:
    :param path:
    :param model_name:
    :param save:
    :return:
    """
    train_l = None
    train_acc = None
    test_acc = None
    metric = None

    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    print('training on', *devices)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    ###########################PID试验区###########################
    rectified_gcc_universal(
        net.module, "dicision",
        softness_ratio=softness_ratio_init,
        rejection_rate=decision_rejection_rates_init,
        compromise_ratio=compromise_ratio_init,
        controlling_unit=controlling_unit)
    pid = PID(Kp, Ki, Kd, clear_threshold, windup_guard)
    print("PID initialized!", f"Kp:{Kp}, Ki:{Ki}, Kd:{Kd}")
    print(f"softness_ratio_init:{softness_ratio_init}")
    print(f"decision_rejection_rates_init:{decision_rejection_rates_init}")
    print(f"compromise_rate_init:{compromise_ratio_init}")
    write_pid_control("#################start################", logfile_name)
    write_pid_control(f"model name:{model_name}", logfile_name)
    write_pid_control(f"lr:{lr}, a_control:{a_control}, b_control:{b_control1}", logfile_name)
    write_pid_control(f"Kp:{Kp}, Ki:{Ki}, Kd:{Kd}", logfile_name)
    write_pid_control(f"clear_threshold:{clear_threshold}, windup_guard:{windup_guard}", logfile_name)
    write_pid_control(
        f"softness_ratio_init:{softness_ratio_init}, decision_rejection_rates_init:{decision_rejection_rates_init}",
        logfile_name)
    write_pid_control(f"compromise_ratio_init:{compromise_ratio_init}", logfile_name)
    ###########################PID试验区###########################
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            y_hat = net(X)
            #########################################################
            # 增加了对激活函数的正则项
            rt1 = rt_for_activation_function_v7(net.module, activation,
                                                a_control,
                                                b_control1, b_control2, b_control3)
            rt2 = rt_for_activation_function_difference(net.module, activation, k1, k2)
            loss_only = loss(y_hat, y)
            l = loss_only + rt1 + rt2
            #########################################################
            l.backward()
            optimizer.step()

            with torch.no_grad():
                # 仅仅使用无rt的loss，避免其影响最终的误差衡量
                metric.add(loss_only * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        ###########################PID试验区###########################
        rejection_rate = pid(train_acc, test_acc) + decision_rejection_rates_init  # 注意；train_acc和test_acc均为浮点数
        rectified_gcc_universal(net.module, "dicision",
                                rejection_rate=rejection_rate, controlling_unit=controlling_unit)
        write_pid_control(f"test:{test_acc}, train:{train_acc}", logfile_name)
        write_pid_control(f"train_loss:{train_l}", logfile_name)
        write_pid_control(">>" + str(rejection_rate), logfile_name)
        ###########################PID试验区###########################
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on ', *devices)

    if save is True:
        train_acc_s = train_acc * 1000
        test_acc_s = test_acc * 1000
        torch.save(
            net.state_dict(),
            path + f"/{model_name}_tr{train_acc_s:.0f}_te{test_acc_s:.0f}"
                   f"_lr{lr}_rt{a_control}_{b_control1}_{b_control2}_wd{weight_decay}.params")

    ###########################PID试验区###########################
    write_pid_control("total time:" + str(timer.sum()), logfile_name)
    write_pid_control("#################end################", logfile_name)
    ###########################PID试验区###########################
    return rejection_rate


def load_parallel_model(path, model):
    state_dict = torch.load(path)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove the "module." prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)
