class Evolver():
    def __init__(self, model):
        self.model = model

    def initialize(self):
        self.it = 0
    
    def step(self, dt):
        raise NotImplementedError("Step is not implemented!")

class Midpoint_Evolver(Evolver):
    def get_method():
        return "Midpoint Integration"
    
    def step(self, dt):
        n1 = self.model.evolution_update(dt / 2, n=self.model.n)
        self.model.n = self.model.evolution_update(dt, n=self.model.n, grad_n=n1)

class Verlet_Evolver(Evolver):
    def get_method():
        return "Verlet Integration"
    
    def initialize(self):
        super().initialize()
        self.n_half = None
    
    def step(self, dt):
        if not self.n_half:
            n1 = self.model.evolution_update(dt / 4, n=self.model.n)
            self.n_half = self.model.evolution_update(dt / 2, n=self.model.n, grad_n=n1)
        self.model.n = self.model.evolution_update(dt, n=self.model.n, grad_n=self.n_half)
        self.n_half = self.model.evolution_update(dt, n=self.n_half, grad_n=self.model.n)
