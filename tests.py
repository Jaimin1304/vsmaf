from models.point import Point
from models.vectorspace import Dimension, VectorSpace
import math


def test():
    print("----- start test -----")
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
    # 序列化和反序列化
    vector_space.to_json("vector_space.json")
    restored_vector_space = VectorSpace.from_json("vector_space.json")
    print("Restored VectorSpace from JSON:")
    print(restored_vector_space)
    print("----- end test -----")


if __name__ == "__main__":
    test()
