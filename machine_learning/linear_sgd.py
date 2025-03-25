import numpy as np

class Linear:
    def __init__(self, input_dim, output_dim):
        self.params = {'w': {}, 'b': {}}
        self.params['w']['data'] = np.random.normal(0, 1, (input_dim, output_dim))
        self.params['b']['data'] = np.random.normal(0, 1, output_dim)
        
    def forward(self, x):
        self.x = x
        self.y = x @ self.params['w']['data'] + self.params['b']['data']
        return self.y
    
    def backward(self, g):
        self.params['w']['grad'] = self.x.T @ g
        self.params['b']['grad'] = g.sum(axis=0)
        return g @ self.params['w']['data'].T
    
    def step(self, lr=1e-4):
        self.params['w']['data'] -= lr * self.params['w']['grad']
        self.params['b']['data'] -= lr * self.params['b']['grad']
    
linear = Linear(64, 1)
x = np.random.normal(0, 1, (4, 64))
y = np.arange(4).reshape(4, 1)

steps = 300
lr = 1e-2
for step in range(steps):
    y_pred = linear.forward(x)
    loss = 0.5 * (y_pred-y)**2 / y.shape[0]
    grad = (y_pred-y) / (y.shape[0])
    linear.backward(grad)
    linear.step(lr=lr)
    print(f"step {step} loss:", loss.mean())
    
print(linear.forward(x))
    