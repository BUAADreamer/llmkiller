import numpy as np

class SGDOptimizer:
    def __init__(self, paramList, lr):
        self.paramList = paramList
        self.lr = lr
        
    def step(self):
        for params in self.paramList:
            for paramName in params:
                params[paramName]['data'] -= params[paramName]['grad'] * self.lr

class Linear:
    def __init__(self, input_dim, output_dim):
        self.params = {'w': {}, 'b': {}}
        self.params['w']['data'] = np.random.normal(0, 1, (input_dim, output_dim))
        self.params['b']['data'] = np.random.normal(0, 1, output_dim)
        
    def forward(self, x):
        """
        y = x @ w.T + b
        """
        self.x = x
        self.y = self.x @ self.params['w']['data'] + self.params['b']['data']
        return self.y
    
    def backward(self, g):
        """
        对于模型层来说，需要两个导数：
        1.链式法则中，对这一层参数求导: 
            Δy/Δw = x.T @ g
            Δy/Δb = g.sum()
        2.链式法则中，对x求导: 
            Δy/Δx = g @ w.T
        """
        self.params['w']['grad'] = self.x.T @ g
        self.params['b']['grad'] = np.sum(g, axis=0)
        return g @ self.params['w']['data'].T

class Activation:
    def __init__(self):
        self.params = {}
        
    def forward(self, x):
        self.x = x
        self.y = self.function(x)
        return self.y
    
    def backward(self, g):
        return self.derivative(self.x) * g
    
class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
    
    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        """
        对于sigmoid而言，只需对x求导
        (0-(-e^-x))/((1+e^-x)^2) 
        = e^-x/((1+e^-x)^2) 
        = e^-x/(1+e^-x) * 1/(1+e^-x) 
        = f * (1-f)
        """
        f = self.function(x)
        return f * (1 - f)

class MLP:
    def __init__(self, input_dim, output_dim, layer=3):
        self.layers = []
        for _ in range(layer-1):
            self.layers.append(Linear(input_dim, input_dim))
            self.layers.append(Sigmoid())
        self.layers.append(Linear(input_dim, output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, g):
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g
    
    def getParamList(self):
        paramList = []
        for layer in self.layers:
            paramList.append(layer.params)
        return paramList

mlp = MLP(64, 1, 3)
optimizer = SGDOptimizer(mlp.getParamList(), 1e-2)
x = np.random.normal(0, 1, (4,64))
for i in range(1,4):
    x[i, :] = x[0, :] + i
print(x)
y = np.arange(4).reshape(4, 1)

steps = 100
for i in range(steps):
    y_pred = mlp.forward(x)
    loss = 0.5 * (y_pred-y)**2 / y.shape[0]
    # 损失函数直接对y_pred求导即可
    grad = (y_pred-y) / y.shape[0]
    mlp.backward(grad)
    optimizer.step()
    print(f"step {i} loss:", loss.mean())

print(mlp.forward(x))
