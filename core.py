# 这是dezero的核心文件，包括核心类和核心函数
# 可以实现求高阶导数的模块

# 导入必要的库

# python内置库
import weakref  # python自带的弱引用
import contextlib

# 第三方库
import numpy as np

# 自建库
import dezero

# 核心类

class Config():
    """反向传播开关"""
    enable_backprop = True
    train = True

class Variable():
    """变量"""
    def __init__(self, data, name=None):
        __array_priority__ = 200
        if data is not None:  # 允许仅创建变量
            if not isinstance(data, np.ndarray):  # 当变量数据类型不是np.ndarray时，提示用户重新输入数据
                raise TypeError('{} is not supported!'.format(type(data)))
            
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output是弱引用

            with using_config('enable_backprop', create_graph):  # 链接到Function的backward，create_graph决定是否允许和当前传播方向相反方向传播路径的建立
                gxs = f.backward(*gys)  # 解包

                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
            
                xs = f.inputs
                for x, gx in zip(xs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx  # 这里是variable实例

                    if x.creator is not None:
                        add_func(x.creator)
            
            if not retain_grad:
                for output in f.outputs:
                    output().grad = None

    def cleargrad(self):
        self.grad = None

    # 把variable包装成ndarray，可以直接使用shape,ndim,dtype这种ndarray专用属性
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def size(self):
        return self.data.size
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    # 实现乘法
    def __mul__(self, other):
        return mul(self, other)
    
    # 实现加法的重载运算符
    def __add__(self, other):
        return add(self, other)
    
    # 实现负数的重载运算符
    def __neg__(self):
        return neg(self)
    
    def __rsub__(self, x1):
        return rsub(self, x1)
    
    def reshape(self, *shape):  # 传入参数可能是2,3这种离散的，所以用*
        """对Variable实例调用shape函数"""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)
    
    def transpose(self):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return dezero.functions.transpose(self, axes)
    
    @property
    def T(self):
        return dezero.functions.transpose(self)
    
    def sum(self, axis=None, keepndims=False):
        return dezero.functions.sum(self, axis, keepndims)

class Function():
    """函数"""
    def __call__(self, *inputs):
        inputs = [as_variable(input) for input in inputs]

        xs = [input.data for input in inputs]
        ys = self.forward(*xs)  # xs是列表
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_ndarray(y)) for y in ys]  # 这里要求ys必须是列表或者元组

        # 反向传播路径建立部分
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        return NotImplementedError
    
    def backward(self, dy):
        return NotImplementedError
    
class Add(Function):
    """加法"""
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape  # 记录传入张量的形状，给反向传播提供形状指导
        y = x0 + x1
        return y
    
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
    
class Mul(Function):
    """乘法"""
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs  # x0是variable，x1是variable
        gx0 = x1 * gy
        gx1 = x0 * gy
        if x0.shape != x1.shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Neg(Function):
    """负数"""
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        gx = -gy
        return gx
    
class Sub(Function):
    """减法"""
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y
    
    def backward(self, gy):
        gx0 = 1 * gy
        gx1 = -1 * gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Div(Function):
    """除法"""
    def forward(self, x0, x1):
        return x0 / x1
    
    def backward(self, gy):
        x0 = self.inputs[0]
        x1 = self.inputs[1]
        gx0 = 1 / x1 * gy
        gx1 = -x0 / x1 ** 2 * gy
        if x0.shape != x1.shape:
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

class Pow(Function):
    """幂运算"""
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx
    
class Parameter(Variable):
    """参数类，直接继承自变量类"""
    pass

# 核心函数

@contextlib.contextmanager
def using_config(name, value):
    """启用反向传播"""
    old_state = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name,  old_state)

def no_grad():
    """禁用反向传播"""
    return using_config('enable_backprop', False)

def as_ndarray(x, array_module=np):
    """将标量转换成ndarray类型"""
    if np.isscalar(x):
        return array_module.array(x)
        
    return x

def as_variable(x):
    """将ndarray转换成变量类型"""
    if not isinstance(x, Variable):
        return Variable(x)
    
    return x

def add(x0, x1):
    """加法函数"""
    x1 = as_ndarray(x1)
    return Add()(x0, x1)

def mul(x0, x1):
    """乘法函数"""
    x1 = as_ndarray(x1)
    return Mul()(x0, x1)

def neg(x):
    """负数函数"""
    return Neg()(x)

def sub(x0, x1):
    """减法函数"""
    x1 = as_ndarray(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    """右减法函数"""
    x1 = as_ndarray(x1)
    return Sub()(x1, x0)

def div(x0, x1):
    """除法函数"""
    x1 = as_ndarray(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    """右除法函数"""
    x1 = as_ndarray(x1)
    return Div()(x1, x0)

def pow(x, c):
    """幂运算函数"""
    return Pow(c)(x)

def setup_variable():
    """运算符重载函数"""
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow

    Variable.__getitem__ = dezero.functions.get_item
    
    Variable.matmul = dezero.functions.matmul
    Variable.dot = dezero.functions.matmul
    Variable.max = dezero.functions.max
    Variable.min = dezero.functions.min