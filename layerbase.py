from abc import abstractmethod

class LayerBase:
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, X, train_mode=True):
        pass

    @abstractmethod
    def forward(self, dLdY):
        pass

    @abstractmethod
    def save_parameters(self):
        pass
    
    @abstractmethod
    def load_parameters(self):
        pass
    
    def __call__(self, X):
        return self.forward(X)

if(__name__=='__main__'):
    pass


