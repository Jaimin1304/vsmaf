import math


class Dimension:
    def __init__(self, name, weight=1.0):
        self.name = name
        self.weight = weight

    def __repr__(self):
        return f"Dimension(name={self.name}, weight={self.weight})"


class VectorSpace:
    def __init__(self, dimensions, dimension_names=None):
        self.dimensions = dimensions
        self.dimension_names = self.initialize_dimension_names(dimension_names)
        self.points = {}

    def initialize_dimension_names(self, dimension_names):
        if dimension_names is None:
            return [str(i) for i in range(self.dimensions)]
        if len(dimension_names) != self.dimensions:
            raise ValueError(
                "Dimension names length does not match the number of dimensions."
            )
        return dimension_names

    def add_point(self, point):
        if len(point.coordinates) != self.dimensions:
            raise ValueError(
                f"Point dimensions ({len(point.coordinates)}) do not match VectorSpace dimensions ({self.dimensions})"
            )
        while point.id in self.points:
            point.id = point.generate_unique_id()
        self.points[point.id] = point

    def remove_point(self, point_id):
        if point_id in self.points:
            del self.points[point_id]
        else:
            raise KeyError(f"No point with id {point_id} found in VectorSpace.")

    def calculate_distance(self, point1_id, point2_id):
        if point1_id not in self.points or point2_id not in self.points:
            raise KeyError("One or both point IDs are not found in VectorSpace.")
        point1 = self.points[point1_id]
        point2 = self.points[point2_id]
        return math.sqrt(
            sum(
                (p1 - p2) ** 2 for p1, p2 in zip(point1.coordinates, point2.coordinates)
            )
        )

    def find_points_within_radius(self, center_point_id, radius):
        if center_point_id not in self.points:
            raise KeyError(f"No point with id {center_point_id} found in VectorSpace.")
        points_within_radius = []
        for point in self.points.values():
            if self.calculate_distance(center_point_id, point.id) <= radius:
                points_within_radius.append(point)
        return points_within_radius

    def __repr__(self):
        return f"VectorSpace(dimensions={self.dimensions}, dimension_names={self.dimension_names}, points={list(self.points.values())})"
