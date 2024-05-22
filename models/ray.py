from models.point import Point


class Ray:
    def __init__(self, start_point, direction_point):
        if len(start_point.coordinates) != len(direction_point.coordinates):
            raise ValueError("Points must have the same number of dimensions.")
        self.start_point = start_point
        self.direction_point = direction_point
        self.direction_vector = [
            dp - sp
            for sp, dp in zip(start_point.coordinates, direction_point.coordinates)
        ]

    def point_at_parameter(self, t):
        """Return a point on the ray given parameter t (t >= 0)"""
        if t < 0:
            raise ValueError("Parameter t must be non-negative for a ray.")
        coordinates = [
            sp + t * dv
            for sp, dv in zip(self.start_point.coordinates, self.direction_vector)
        ]
        return Point(name=f"Point_on_ray_{t}", coordinates=coordinates)

    def __repr__(self):
        return f"Ray(start_point={self.start_point}, direction_point={self.direction_point})"
