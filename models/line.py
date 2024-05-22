from models.point import Point


class Line:
    def __init__(self, point1, point2):
        if len(point1.coordinates) != len(point2.coordinates):
            raise ValueError("Points must have the same number of dimensions.")
        self.point1 = point1
        self.point2 = point2

    def point_at_parameter(self, t):
        """Return a point on the line given parameter t (0 <= t <= 1)"""
        coordinates = [
            p1 + t * (p2 - p1)
            for p1, p2 in zip(self.point1.coordinates, self.point2.coordinates)
        ]
        return Point(name=f"Point_on_line_{t}", coordinates=coordinates)

    def __repr__(self):
        return f"Line(point1={self.point1}, point2={self.point2})"
