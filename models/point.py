import math
import uuid


class Point:
    def __init__(self, name=None, coordinates=None, labels={}):
        self.id = self.generate_unique_id()
        self.name = name if name else f"Point_{self.id}"
        self.coordinates = coordinates if coordinates else []
        self.length = self.calculate_length()
        self.labels = labels

    def generate_unique_id(self):
        return str(uuid.uuid4())

    def calculate_length(self):
        return math.sqrt(sum(coord**2 for coord in self.coordinates))

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "coordinates": [float(coord) for coord in self.coordinates],
            "labels": self.labels,
        }

    @staticmethod
    def from_dict(data):
        point = Point(
            name=data["name"], coordinates=data["coordinates"], labels=data["labels"]
        )
        point.id = data["id"]
        return point

    def __repr__(self):
        return f"{self.name}(id={self.id}, coordinates={self.coordinates}, length={self.length})"
