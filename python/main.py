import uuid
import math
import numpy as np


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
        return f"{self.name}(id={self.id}, coordinates={self.coordinates}, length={self.length})"


class Dimension:
    def __init__(self, name, weight=1.0):
        self.name = name
        self.weight = weight

    def __repr__(self):
        return f"Dimension(name={self.name}, weight={self.weight})"


class VectorSpace:
    def __init__(self, dimensions):
        if not dimensions:
            raise ValueError("At least one dimension is required.")
        self.dimensions = dimensions
        self.dimension_names = [dim.name for dim in dimensions]
        self.points = {}

    def add_point(self, point):
        if len(point.coordinates) != len(self.dimensions):
            raise ValueError(
                f"Point dimensions ({len(point.coordinates)}) do not match VectorSpace dimensions ({len(self.dimensions)})"
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
                (p1 - p2) ** 2 * dim.weight
                for p1, p2, dim in zip(
                    point1.coordinates, point2.coordinates, self.dimensions
                )
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

    def sort_points_by_dimension(self, dimension, ascending=True):
        if isinstance(dimension, str):
            dimension_index = next(
                (
                    index
                    for index, dim in enumerate(self.dimensions)
                    if dim.name == dimension
                ),
                None,
            )
            if dimension_index is None:
                raise ValueError(f"Dimension name {dimension} not found.")
        elif isinstance(dimension, int):
            if dimension < 0 or dimension >= len(self.dimensions):
                raise ValueError(f"Dimension index {dimension} out of range.")
            dimension_index = dimension
        else:
            raise TypeError("Dimension must be a string or an integer.")
        sorted_points = sorted(
            self.points.values(),
            key=lambda point: point.coordinates[dimension_index],
            reverse=not ascending,
        )
        return sorted_points

    def __repr__(self):
        return f"VectorSpace(dimensions={self.dimensions}, points={list(self.points.values())})"


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


def main():
    print("----- start main -----")
    # 创建一些 Dimension 实例
    dimension_x = Dimension(name="X", weight=1.0)
    dimension_y = Dimension(name="Y", weight=1.0)
    dimension_z = Dimension(name="Z", weight=1.0)
    dimensions = [dimension_x, dimension_y, dimension_z]

    # 创建一些 Point 实例
    point1 = Point(name="Point1", coordinates=[1.0, 2.0, 3.0])
    point2 = Point(name="Point2", coordinates=[4.0, 5.0, 6.0])
    point3 = Point(coordinates=[7.0, 8.0, 9.0])  # 自动生成名称

    # 创建一个 VectorSpace 实例，定义维度
    vector_space = VectorSpace(dimensions)

    # 添加 Point 到 VectorSpace 中
    vector_space.add_point(point1)
    vector_space.add_point(point2)
    vector_space.add_point(point3)

    # 打印 VectorSpace 以验证
    print(vector_space)

    # 计算两个点之间的距离
    distance = vector_space.calculate_distance(point1.id, point2.id)
    print(f"Distance between {point1.name} and {point2.name} is {distance}")

    # 找出以 point1 为中心，半径为 5 的高维球体内的所有点
    radius = 5.0
    points_within_radius = vector_space.find_points_within_radius(point1.id, radius)
    print(f"Points within radius {radius} of {point1.name}: {points_within_radius}")

    # 按照维度 "Y" 进行升序排序
    sorted_points = vector_space.sort_points_by_dimension("Y", ascending=True)
    print(f"Points sorted by dimension 'Y' (ascending): {sorted_points}")

    # 移除一个 Point
    vector_space.remove_point(point1.id)

    # 打印 VectorSpace 以验证
    print(vector_space)

    print("----- end main -----")


if __name__ == "__main__":
    main()
