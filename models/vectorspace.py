import csv
import uuid
import math
import json
import vispy
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from dimension import Dimension
from point import Point

vispy.use("pyqt6")


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
        # labels = kmeans.labels_
        labels = [int(label) for label in kmeans.labels_]
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
        # labels = dbscan.labels_
        labels = [int(label) for label in dbscan.labels_]
        # Assign cluster labels to points
        for point, label in zip(self.points.values(), labels):
            point.labels["dbscan_clusters"] = label
        return labels

    def to_json(self, filepath):
        data = {
            "dimensions": [dim.to_dict() for dim in self.dimensions],
            "points": [point.to_dict() for point in self.points.values()],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def from_json(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        dimensions = [Dimension.from_dict(dim_data) for dim_data in data["dimensions"]]
        vector_space = VectorSpace(dimensions)
        for point_data in data["points"]:
            point = Point.from_dict(point_data)
            vector_space.add_point(point)
        return vector_space

    @staticmethod
    def minmax(coords):
        scaler = MinMaxScaler()
        return scaler.fit_transform(coords)

    @staticmethod
    def zscore(coords):
        scaler = StandardScaler()
        return scaler.fit_transform(coords)

    @staticmethod
    def robust(coords):
        scaler = RobustScaler()
        return scaler.fit_transform(coords)

    @staticmethod
    def from_csv(
        filepath,
        dimension_indices,
        name_index=None,
        id_index=None,
        label_indices=None,
        delimiter=",",
    ):
        """
        Create a VectorSpace from a CSV file.
        Parameters:
        filepath (str): Path to the CSV file.
        dimension_indices (list of int): Indices of the columns to be used as dimensions.
        name_index (int): Index of the column to be used as point names. If None, names will be autogenerated.
        id_index (int): Index of the column to be used as point IDs. If None, IDs will be autogenerated.
        label_indices (list of int): Indices of the columns to be used as point labels. Default is None.
        delimiter (str): The delimiter of the CSV file. Default is ','.
        Returns:
        VectorSpace: The created VectorSpace instance.
        """
        with open(filepath, "r") as file:
            reader = csv.reader(file, delimiter=delimiter)
            header = next(reader)
            dimensions = [
                Dimension(name=header[i], weight=1.0) for i in dimension_indices
            ]
            data = list(reader)
            # Extract and normalize coordinates in dimension_indices
            coords = np.array(
                [[float(row[i]) for i in dimension_indices] for row in data]
            )
            normalized_coords = VectorSpace.minmax(coords)
            # normalized_coords = VectorSpace.zscore(coords)
            # normalized_coords = VectorSpace.robust(coords)
            vector_space = VectorSpace(dimensions)
            for i, row in enumerate(data):
                coordinates = normalized_coords[i].tolist()
                name = row[name_index] if name_index is not None else None
                id = row[id_index] if id_index is not None else None
                # Extract labels
                labels = {}
                if label_indices is not None:
                    for index in label_indices:
                        labels[header[index]] = row[index]
                point = Point(id=id, name=name, coordinates=coordinates, labels=labels)
                vector_space.add_point(point)
        return vector_space

    def __repr__(self):
        return f"VectorSpace(dimensions={self.dimensions}, points={list(self.points.values())})"


if __name__ == "__main__":
    print("vectorspace.py")
