import math
import uuid


class Point:
    def __init__(self, name=None, coordinates=None):
        self.id = self.generate_unique_id()
        self.name = name if name else f"Point_{self.id}"
        self.coordinates = coordinates if coordinates else []
        self.length = self.calculate_length()

    def generate_unique_id(self):
        return str(uuid.uuid4())

    def calculate_length(self):
        return math.sqrt(sum(coord**2 for coord in self.coordinates))

    def __repr__(self):
        return f"Point(id={self.id}, name='{self.name}', coordinates={self.coordinates}, length={self.length})"
