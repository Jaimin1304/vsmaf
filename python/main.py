import uuid
import math


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

    def sort_points_by_dimension(self, dimension, ascending=True):
        if isinstance(dimension, str):
            if dimension not in self.dimension_names:
                raise ValueError(f"Dimension name {dimension} not found.")
            dimension_index = self.dimension_names.index(dimension)
        elif isinstance(dimension, int):
            if dimension < 0 or dimension >= self.dimensions:
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
        return f"VectorSpace(dimensions={self.dimensions}, dimension_names={self.dimension_names}, points={list(self.points.values())})"


def main():
    print("----- start main -----")
    # 创建一些 Point 实例
    point1 = Point(name="Point1", coordinates=[1.0, 2.0, 3.0])
    point2 = Point(name="Point2", coordinates=[4.0, 5.0, 6.0])
    point3 = Point(coordinates=[7.0, 8.0, 9.0])  # 自动生成名称
    # 创建一个 3 维的 VectorSpace 实例，定义维度名称
    vector_space = VectorSpace(3, dimension_names=["X", "Y", "Z"])
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
