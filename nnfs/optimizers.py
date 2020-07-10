

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def __call__(self, params, grad_param):
        return params - self.lr * grad_param
