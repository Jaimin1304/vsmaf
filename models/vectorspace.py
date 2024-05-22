import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import vispy

vispy.use("pyqt6")
import matplotlib.pyplot as plt
from vispy import scene
from vispy.scene import visuals


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

    def pca_transform(self, n_components):
        """
        Perform PCA on the points in the vector space and return the transformed coordinates.
        Parameters:
        n_components (int): Number of dimensions to reduce to.
        Returns:
        np.ndarray: Transformed coordinates.
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
        return transformed_data

    def tsne_transform(self, n_components=2, perplexity=2.0, max_iter=1000):
        """
        Perform t-SNE on the points in the vector space and return the transformed coordinates.
        Parameters:
        n_components (int): Number of dimensions to reduce to.
        perplexity (float): The perplexity parameter for t-SNE.
        max_iter (int): The number of iterations for optimization.
        Returns:
        np.ndarray: Transformed coordinates.
        """
        if n_components > len(self.dimensions):
            raise ValueError(
                "n_components cannot be greater than the number of dimensions in the vector space."
            )
        # Collect coordinates of all points
        data = np.array([point.coordinates for point in self.points.values()])
        if perplexity >= len(data):
            print(data)
            raise ValueError("perplexity must be less than the number of samples!")
        # Perform t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, max_iter=max_iter)
        transformed_data = tsne.fit_transform(data)
        return transformed_data

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
