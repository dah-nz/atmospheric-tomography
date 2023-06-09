class Basis():
    def __init__(self, name=""):
        self.name = name
    def out(self, grid, order):
        raise NotImplementedError()
    