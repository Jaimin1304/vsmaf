import numpy as np
from models.point import Point


class Plane:
    def __init__(self, points, plane_dimension):
        if len(points) < 2:
            raise ValueError("At least two points are required to define a plane.")
        self.points = points
        self.space_dimension = len(points[0].coordinates)
        if any(len(point.coordinates) != self.space_dimension for point in points):
            raise ValueError("All points must have the same number of dimensions.")
        if plane_dimension >= self.space_dimension:
            raise ValueError(
                f"Plane dimension must be less than the space dimension ({self.space_dimension})."
            )
        if len(points) != plane_dimension + 1:
            raise ValueError(
                f"Exactly {plane_dimension + 1} points are required to define a {plane_dimension}-dimensional plane in {self.space_dimension}-dimensional space."
            )
        self.plane_dimension = plane_dimension
        if not self._check_points_are_coplanar():
            raise ValueError("The provided points are not coplanar.")

    def _check_points_are_coplanar(self):
        """Check if the provided points are coplanar by using the determinant method."""
        vectors = [
            np.array(point.coordinates) - np.array(self.points[0].coordinates)
            for point in self.points[1:]
        ]
        matrix = np.stack(vectors, axis=0)
        return np.linalg.matrix_rank(matrix) == self.plane_dimension

    def point_belongs_to_plane(self, point):
        """Check if a given point belongs to the plane."""
        vectors = [
            np.array(p.coordinates) - np.array(self.points[0].coordinates)
            for p in self.points[1:]
        ] + [np.array(point.coordinates) - np.array(self.points[0].coordinates)]
        matrix = np.stack(vectors, axis=0)
        return np.linalg.matrix_rank(matrix) == self.plane_dimension

    def __repr__(self):
        return f"Plane(points={self.points}, plane_dimension={self.plane_dimension})"


# 示例用法
if __name__ == "__main__":
    # 创建一些 Point 实例
    point1 = Point(name="Point1", coordinates=[1.0, 2.0, 3.0])
    point2 = Point(name="Point2", coordinates=[4.0, 5.0, 6.0])
    point3 = Point(name="Point3", coordinates=[7.0, 8.0, 9.0])
    point4 = Point(name="Point4", coordinates=[10.0, 11.0, 12.0])

    # 创建一个 2 维的 Plane 实例（在 3 维空间中）
    plane = Plane([point1, point2, point3], plane_dimension=2)
    print(plane)

    # 检查一个点是否在平面上
    print(f"Does point4 belong to the plane? {plane.point_belongs_to_plane(point4)}")
