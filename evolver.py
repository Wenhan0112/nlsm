
class Evolver():
    def __init__(self, params, step_size: float):
        """
        Constructor
        """
        assert step_size > 0
        self.step_size = step_size
        self.params = params
        
    
    def step(self):
        raise NotImplementedError("Step is not implemented!")


