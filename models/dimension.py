class Dimension:
    def __init__(self, name, weight=1.0):
        self.name = name
        self.weight = weight

    def to_dict(self):
        return {"name": self.name, "weight": self.weight}

    @staticmethod
    def from_dict(data):
        return Dimension(name=data["name"], weight=data["weight"])

    def __repr__(self):
        return f"Dimension(name={self.name}, weight={self.weight})"
