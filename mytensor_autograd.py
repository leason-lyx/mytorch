import numpy as np
import mytensor


class Context:
    def __init__(self):
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class Function:
    @classmethod
    def apply(cls, *args):
        ctx = Context()
        raw_args = []
        requires_grad = False
        device = "cpu"
        for arg in args:
            if isinstance(arg, Tensor):
                raw_args.append(arg.data)
                requires_grad = requires_grad or arg.requires_grad
                device = arg.device
            else:
                raw_args.append(arg)
        out_data = cls.forward(ctx, *raw_args)
        out = Tensor(out_data, requires_grad=requires_grad, device=device)
        if requires_grad:
            ctx.inputs = args
            ctx.op = cls
            out._ctx = ctx
        return out


class Tensor:
    def __init__(self, data, requires_grad=False, device="cpu"):
        self.requires_grad = requires_grad
        self.grad = None
        self.device = device
        self._ctx = None
        if isinstance(data, mytensor.Tensor):
            self.data = data
        else:
            np_data = np.array(data, dtype=np.float32)
            self.data = mytensor.Tensor(np_data, mytensor.Device(device))

    @property
    def shape(self):
        return list(self.data.shape())

    def numpy(self):
        return self.data.numpy()

    def detach(self):
        return Tensor(self.data, requires_grad=False, device=self.device)

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = Tensor(np.ones(self.shape, dtype=np.float32), device=self.device)
        elif not isinstance(grad, Tensor):
            grad = Tensor(grad, device=self.device)

        topo = []
        visited = set()

        def build(node):
            if node in visited:
                return
            visited.add(node)
            if node._ctx is not None:
                for parent in node._ctx.inputs:
                    if isinstance(parent, Tensor):
                        build(parent)
            topo.append(node)

        build(self)
        node_to_grads = {self: [grad]}

        for node in reversed(topo):
            grads = node_to_grads.get(node, [])
            if not grads:
                continue
            node.grad = _sum_grads(grads, node.device)
            if node._ctx is None:
                continue
            input_grads = node._ctx.op.backward(node._ctx, node.grad)
            if not isinstance(input_grads, tuple):
                input_grads = (input_grads,)
            for inp, grad_inp in zip(node._ctx.inputs, input_grads):
                if not isinstance(inp, Tensor) or not inp.requires_grad:
                    continue
                if grad_inp is None:
                    continue
                node_to_grads.setdefault(inp, []).append(grad_inp)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, device={self.device})"


class Parameter(Tensor):
    def __init__(self, data, device="cpu"):
        super().__init__(data, requires_grad=True, device=device)


class Module:
    def parameters(self):
        params = []
        for value in self.__dict__.values():
            params.extend(_collect_params(value))
        return params

    def zero_grad(self):
        for param in self.parameters():
            param.grad = None


def _collect_params(obj):
    if isinstance(obj, Parameter):
        return [obj]
    if isinstance(obj, Module):
        return obj.parameters()
    if isinstance(obj, (list, tuple)):
        params = []
        for item in obj:
            params.extend(_collect_params(item))
        return params
    if isinstance(obj, dict):
        params = []
        for item in obj.values():
            params.extend(_collect_params(item))
        return params
    return []


def _sum_grads(grads, device):
    if len(grads) == 1:
        return grads[0]
    total = grads[0].data.numpy()
    for grad in grads[1:]:
        total = total + grad.data.numpy()
    return Tensor(total, device=device)


def _scale_tensor(tensor, scale):
    data = tensor.data.numpy() * scale
    return Tensor(data, device=tensor.device)


class Conv2dFn(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride_h, stride_w, pad_h, pad_w):
        out = mytensor.conv2d(input, weight, bias, stride_h, stride_w, pad_h, pad_w)
        ctx.save_for_backward(input, out, weight, bias)
        ctx.stride = (stride_h, stride_w)
        ctx.padding = (pad_h, pad_w)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, out, weight, bias = ctx.saved_tensors
        stride_h, stride_w = ctx.stride
        pad_h, pad_w = ctx.padding
        grad_input, grad_weight, grad_bias = mytensor.conv2d_backward(
            input, out, weight, bias, stride_h, stride_w, pad_h, pad_w, grad_output.data
        )
        device = grad_output.device
        return (
            Tensor(grad_input, device=device),
            Tensor(grad_weight, device=device),
            Tensor(grad_bias, device=device),
            None,
            None,
            None,
            None,
        )


class LinearFn(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        out = mytensor.linear(input, weight, bias)
        ctx.save_for_backward(input, weight, bias)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = mytensor.linear_backward(
            input, weight, bias, grad_output.data
        )
        device = grad_output.device
        return (
            Tensor(grad_input, device=device),
            Tensor(grad_weight, device=device),
            Tensor(grad_bias, device=device),
        )


class ReLUFn(Function):
    @staticmethod
    def forward(ctx, input):
        out = mytensor.relu(input)
        ctx.save_for_backward(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = mytensor.relu_backward(input, grad_output.data)
        return Tensor(grad_input, device=grad_output.device)


class MaxPool2dFn(Function):
    @staticmethod
    def forward(ctx, input, pool_h, pool_w, stride_h, stride_w):
        out = mytensor.max_pool2d(input, pool_h, pool_w, stride_h, stride_w)
        ctx.save_for_backward(input, out)
        ctx.pool = (pool_h, pool_w)
        ctx.stride = (stride_h, stride_w)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, out = ctx.saved_tensors
        pool_h, pool_w = ctx.pool
        stride_h, stride_w = ctx.stride
        grad_input = mytensor.max_pool2d_backward(
            input, out, pool_h, pool_w, stride_h, stride_w, grad_output.data
        )
        return Tensor(grad_input, device=grad_output.device), None, None, None, None


class ReshapeFn(Function):
    @staticmethod
    def forward(ctx, input, shape):
        ctx.original_shape = list(input.shape())
        out = input.reshape([int(dim) for dim in shape])
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.data.reshape(ctx.original_shape)
        return Tensor(grad_input, device=grad_output.device), None


class CrossEntropyLossFn(Function):
    @staticmethod
    def forward(ctx, logits, labels):
        ctx.save_for_backward(logits)
        ctx.labels = labels
        return mytensor.cross_entropy_loss(logits, labels)

    @staticmethod
    def backward(ctx, grad_output):
        (logits,) = ctx.saved_tensors
        grad_logits = mytensor.cross_entropy_loss_backward(logits, ctx.labels)
        grad = Tensor(grad_logits, device=grad_output.device)
        scale = float(grad_output.data.numpy().reshape(-1)[0])
        if scale != 1.0:
            grad = _scale_tensor(grad, scale)
        return grad, None


def conv2d(x, weight, bias, stride=1, padding=0):
    stride_h = stride_w = int(stride)
    pad_h = pad_w = int(padding)
    return Conv2dFn.apply(x, weight, bias, stride_h, stride_w, pad_h, pad_w)


def linear(x, weight, bias):
    return LinearFn.apply(x, weight, bias)


def relu(x):
    return ReLUFn.apply(x)


def max_pool2d(x, kernel_size, stride=None):
    pool_h = pool_w = int(kernel_size)
    stride_val = int(kernel_size if stride is None else stride)
    return MaxPool2dFn.apply(x, pool_h, pool_w, stride_val, stride_val)


def reshape(x, shape):
    return ReshapeFn.apply(x, shape)


def flatten(x):
    shape = x.shape
    batch = shape[0]
    rest = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    return reshape(x, (batch, rest))


def cross_entropy_loss(logits, labels):
    labels_np = np.asarray(labels, dtype=np.int32)
    return CrossEntropyLossFn.apply(logits, labels_np)


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, device="cpu"):
        self.stride = int(stride)
        self.padding = int(padding)
        k = int(kernel_size)
        scale = np.sqrt(2.0 / (in_channels * k * k))
        weight = np.random.randn(out_channels, in_channels, k, k).astype(np.float32) * scale
        bias = np.zeros((out_channels,), dtype=np.float32)
        self.weight = Parameter(weight, device=device)
        self.bias = Parameter(bias, device=device)

    def __call__(self, x):
        return conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


class Linear(Module):
    def __init__(self, in_features, out_features, device="cpu"):
        scale = np.sqrt(2.0 / in_features)
        weight = np.random.randn(in_features, out_features).astype(np.float32) * scale
        bias = np.zeros((out_features,), dtype=np.float32)
        self.weight = Parameter(weight, device=device)
        self.bias = Parameter(bias, device=device)

    def __call__(self, x):
        return linear(x, self.weight, self.bias)


class ReLU(Module):
    def __call__(self, x):
        return relu(x)


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = int(kernel_size)
        self.stride = int(kernel_size if stride is None else stride)

    def __call__(self, x):
        return max_pool2d(x, self.kernel_size, self.stride)


class Flatten(Module):
    def __call__(self, x):
        return flatten(x)


class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    def step(self):
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = {}

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            w = param.data.numpy()
            g = param.grad.data.numpy()
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * w
            if self.momentum != 0.0:
                v = self._velocity.get(param)
                if v is None:
                    v = np.zeros_like(g)
                v = self.momentum * v + g
                self._velocity[param] = v
                update = v
            else:
                update = g
            w = w - self.lr * update
            param.data = mytensor.Tensor(w.astype(np.float32), mytensor.Device(param.device))


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = {}
        self._v = {}
        self._t = 0

    def step(self):
        self._t += 1
        for param in self.params:
            if param.grad is None:
                continue
            w = param.data.numpy()
            g = param.grad.data.numpy()
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * w
            m = self._m.get(param)
            v = self._v.get(param)
            if m is None:
                m = np.zeros_like(g)
                v = np.zeros_like(g)
            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * (g * g)
            self._m[param] = m
            self._v[param] = v
            m_hat = m / (1.0 - self.beta1 ** self._t)
            v_hat = v / (1.0 - self.beta2 ** self._t)
            w = w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            param.data = mytensor.Tensor(w.astype(np.float32), mytensor.Device(param.device))
