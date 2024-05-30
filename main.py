import uuid
import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
import vispy

vispy.use("pyqt6")
import matplotlib.pyplot as plt
from vispy import scene
from vispy.scene import visuals


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

    def filter_points_by_ranges(self, ranges):
        """
        Filter points based on the specified ranges for each dimension.
        Parameters:
        ranges (list of list of float): A nested list where each inner list contains two floats
                                        representing the interval [min, max] for the corresponding dimension.
        Returns:
        list of Point: The points that fall within the specified ranges for all dimensions.
        """
        # 检查 ranges 的长度是否与 dimensions 的数量一致
        if len(ranges) != len(self.dimensions):
            raise ValueError(
                "The size of ranges must match the number of dimensions in the VectorSpace."
            )
        # 检查每个区间是否有效
        for range_ in ranges:
            if len(range_) != 2:
                raise ValueError(
                    "Each range must be a list of two elements representing [min, max]."
                )
            min_val, max_val = range_
            if min_val > max_val:
                raise ValueError(
                    f"Invalid range: [{min_val}, {max_val}]. Min value cannot be greater than max value."
                )
        filtered_points = []
        for point in self.points.values():
            include_point = True
            for dim_index, (min_val, max_val) in enumerate(ranges):
                if not (min_val <= point.coordinates[dim_index] <= max_val):
                    include_point = False
                    break
            if include_point:
                filtered_points.append(point)
        return filtered_points

    def pca_transform(self, n_components, return_space=False):
        """
        Perform PCA on the points in the vector space and return the transformed coordinates or a new low-dimensional VectorSpace.
        Parameters:
        n_components (int): Number of dimensions to reduce to.
        return_space (bool): Whether to return a new low-dimensional VectorSpace. Default is False.
        Returns:
        np.ndarray or VectorSpace: Transformed coordinates or a new low-dimensional VectorSpace.
        """
        if n_components > len(self.dimensions):
            raise ValueError(
                "n_components cannot be greater than the number of dimensions in the vector space."
            )
        # Collect coordinates of all points
        data = np.array([point.coordinates for point in self.points.values()])
        # Perform PCA
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(data)
        if not return_space:
            return transformed_data
        # Create new dimensions with random names and weight 1.0 for the low-dimensional VectorSpace
        new_dimensions = [
            Dimension(name=f"Dim_{i+1}", weight=1.0) for i in range(n_components)
        ]
        new_space = VectorSpace(new_dimensions)
        # Add transformed points to the new VectorSpace
        for point, new_coords in zip(self.points.values(), transformed_data):
            new_point = Point(name=point.name, coordinates=list(new_coords))
            new_point.id = point.id  # Retain the original ID
            new_space.add_point(new_point)
        return new_space

    def tsne_transform(
        self, n_components=2, perplexity=2.0, max_iter=1000, return_space=False
    ):
        """
        Perform t-SNE on the points in the vector space and return the transformed coordinates or a new low-dimensional VectorSpace.
        Parameters:
        n_components (int): Number of dimensions to reduce to.
        perplexity (float): The perplexity parameter for t-SNE.
        max_iter (int): The number of iterations for optimization.
        return_space (bool): Whether to return a new low-dimensional VectorSpace. Default is False.
        Returns:
        np.ndarray or VectorSpace: Transformed coordinates or a new low-dimensional VectorSpace.
        """
        if n_components > len(self.dimensions):
            raise ValueError(
                "n_components cannot be greater than the number of dimensions in the vector space."
            )
        # Collect coordinates of all points
        data = np.array([point.coordinates for point in self.points.values()])
        if perplexity >= len(data):
            raise ValueError("perplexity must be less than the number of samples!")
        # Perform t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, max_iter=max_iter)
        transformed_data = tsne.fit_transform(data)
        if not return_space:
            return transformed_data
        # Create new dimensions with random names and weight 1.0 for the low-dimensional VectorSpace
        new_dimensions = [
            Dimension(name=f"Dim_{i+1}", weight=1.0) for i in range(n_components)
        ]
        new_space = VectorSpace(new_dimensions)
        # Add transformed points to the new VectorSpace
        for point, new_coords in zip(self.points.values(), transformed_data):
            new_point = Point(name=point.name, coordinates=list(new_coords))
            new_point.id = point.id  # Retain the original ID
            new_space.add_point(new_point)
        return new_space

    def perform_kmeans(self, n_clusters):
        """
        Perform K-means clustering on the points in the vector space.
        Parameters:
        n_clusters (int): The number of clusters to form.
        Returns:
        list: A list of cluster labels for each point.
        """
        if n_clusters <= 0:
            raise ValueError("Number of clusters must be a positive integer.")
        # Collect coordinates of all points
        data = np.array([point.coordinates for point in self.points.values()])
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(data)
        labels = kmeans.labels_
        # Assign cluster labels to points
        for point, label in zip(self.points.values(), labels):
            point.labels["keans_clusters"] = label
        return labels

    def perform_dbscan(self, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering on the points in the vector space.
        Parameters:
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        Returns:
        list: A list of cluster labels for each point.
        """
        # Collect coordinates of all points
        data = np.array([point.coordinates for point in self.points.values()])
        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(data)
        labels = dbscan.labels_
        # Assign cluster labels to points
        for point, label in zip(self.points.values(), labels):
            point.labels["dbscan_clusters"] = label
        return labels

    def visualize_2d(self, dimension_indices=None):
        """
        Visualize the points in 2D space using the specified dimensions.
        Parameters:
        dimension_indices (list of int): Indices of the dimensions to visualize. Default is the first two dimensions.
        """
        if dimension_indices is None:
            dimension_indices = [0, 1]
        if len(dimension_indices) != 2:
            raise ValueError("Two dimensions must be specified for 2D visualization.")
        if any(index >= len(self.dimensions) for index in dimension_indices):
            raise ValueError("Dimension index out of range.")
        fig, ax = plt.subplots()
        # 绘制原点
        ax.scatter(0, 0, color="blue", s=100)  # 原点用蓝色大点表示
        ax.text(0, 0, "Origin", color="blue", fontsize=12, ha="right")
        for point in self.points.values():
            ax.scatter(
                point.coordinates[dimension_indices[0]],
                point.coordinates[dimension_indices[1]],
                label=point.name,
            )
            ax.text(
                point.coordinates[dimension_indices[0]],
                point.coordinates[dimension_indices[1]],
                point.name,
                fontsize=9,
                ha="right",
            )
        ax.set_xlabel(self.dimensions[dimension_indices[0]].name)
        ax.set_ylabel(self.dimensions[dimension_indices[1]].name)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(color="gray", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.show()

    def visualize_3d(self, dimension_indices=None):
        """
        Visualize the points in 3D space using the specified dimensions.
        Parameters:
        dimension_indices (list of int): Indices of the dimensions to visualize. Default is the first three dimensions.
        """
        if dimension_indices is None:
            dimension_indices = [0, 1, 2]
        if len(dimension_indices) != 3:
            raise ValueError("Three dimensions must be specified for 3D visualization.")
        if any(index >= len(self.dimensions) for index in dimension_indices):
            raise ValueError("Dimension index out of range.")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # 绘制原点
        ax.scatter(0, 0, 0, color="blue", s=100, label="Origin")  # 原点用蓝色大点表示
        ax.text(0, 0, 0, "Origin", color="blue", fontsize=12, ha="right")
        for point in self.points.values():
            ax.scatter(
                point.coordinates[dimension_indices[0]],
                point.coordinates[dimension_indices[1]],
                point.coordinates[dimension_indices[2]],
                label=point.name,
            )
            ax.text(
                point.coordinates[dimension_indices[0]],
                point.coordinates[dimension_indices[1]],
                point.coordinates[dimension_indices[2]],
                point.name,
            )
        ax.set_xlabel(self.dimensions[dimension_indices[0]].name)
        ax.set_ylabel(self.dimensions[dimension_indices[1]].name)
        ax.set_zlabel(self.dimensions[dimension_indices[2]].name)
        plt.legend()
        plt.show()

    def visualize_3d_vispy(self, dimension_indices=None):
        """
        Visualize the points in 3D space using the specified dimensions with GPU acceleration.
        Parameters:
        dimension_indices (list of int): Indices of the dimensions to visualize. Default is the first three dimensions.
        """
        if dimension_indices is None:
            dimension_indices = [0, 1, 2]
        if len(dimension_indices) != 3:
            raise ValueError("Three dimensions must be specified for 3D visualization.")
        if any(index >= len(self.dimensions) for index in dimension_indices):
            raise ValueError("Dimension index out of range.")
        # 创建一个场景画布
        canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="white")
        view = canvas.central_widget.add_view()
        view.camera = "turntable"  # 设置相机类型
        # 创建散点
        scatter = visuals.Markers()
        points = np.array([point.coordinates for point in self.points.values()])
        scatter.set_data(points[:, dimension_indices], face_color="red", size=10)
        # 在每个点上显示名称
        for point in self.points.values():
            text = visuals.Text(
                text=point.name,
                color="black",
                anchor_x="left",
                anchor_y="bottom",
                font_size=10,
            )
            text.pos = (
                point.coordinates[dimension_indices[0]],
                point.coordinates[dimension_indices[1]],
                point.coordinates[dimension_indices[2]],
            )
            view.add(text)
        view.add(scatter)
        # 添加坐标轴
        axis = visuals.XYZAxis(parent=view.scene)
        # 显示原点
        origin = visuals.Markers()
        origin.set_data(np.array([[0, 0, 0]]), face_color="blue", size=10)
        view.add(origin)
        # 为坐标轴添加标签
        axis_labels = {
            self.dimensions[dimension_indices[0]].name: (1, 0, 0),
            self.dimensions[dimension_indices[1]].name: (0, 1, 0),
            self.dimensions[dimension_indices[2]].name: (0, 0, 1),
        }
        for label, pos in axis_labels.items():
            text = visuals.Text(
                text=label,
                color="black",
                anchor_x="center",
                anchor_y="center",
                font_size=15,
                bold=True,
            )
            text.pos = np.array(pos) * 1.1  # 让标签稍微远离原点
            view.add(text)
        canvas.app.run()

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
    # Perform PCA
    pca_result = vector_space.pca_transform(n_components=2, return_space=True)
    print("PCA Result as VectorSpace:")
    print(pca_result)
    # Perform t-SNE
    tsne_result = vector_space.tsne_transform(
        n_components=2, perplexity=2, return_space=True
    )
    print("t-SNE Result as VectorSpace:")
    print(tsne_result)
    # 使用区间筛选点
    ranges = [[0, 5], [1, 6], [2, math.inf]]
    filtered_points = vector_space.filter_points_by_ranges(ranges)
    print(f"Filtered points within ranges {ranges}: {filtered_points}")
    # 可视化 3D
    vector_space.visualize_3d()
    # 可视化 2D
    vector_space.visualize_2d()
    # Perform K-means clustering with PCA
    n_clusters = 2
    kmeans_labels = vector_space.perform_kmeans(n_clusters)
    print(f"K-means clustering result with {n_clusters} clusters: {kmeans_labels}")
    # Perform DBSCAN clustering
    eps = 1.0
    min_samples = 2
    dbscan_labels = vector_space.perform_dbscan(eps=eps, min_samples=min_samples)
    print(
        f"DBSCAN clustering result with eps={eps} and min_samples={min_samples}: {dbscan_labels}"
    )
    # 移除一个 Point
    vector_space.remove_point(point1.id)
    print("----- end main -----")


if __name__ == "__main__":
    main()
