import math
from visualizer import visualize_2d, visualize_3d, visualize_3d_vispy
from models.vectorspace import VectorSpace

print("----- run -----")
filepath = "nodes.csv"
dimension_indices = [5, 6, 8]
name_index = 9
id_index = 0
label_indices = None
delimiter = ","
scaling_methods = {
    0: "minmax_invert",
    1: "minmax",
    2: "minmax",
}
vector_space = VectorSpace.from_csv(
    filepath,
    dimension_indices,
    name_index,
    id_index,
    label_indices,
    delimiter,
    scaling_methods,
)

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

# Perform K-means clustering with PCA
n_clusters = 5
kmeans_labels = vector_space.perform_kmeans(n_clusters)
print(f"K-means clustering result with {n_clusters} clusters: {kmeans_labels}")

# 可视化 3D
visualize_3d_vispy(vector_space)

# Perform DBSCAN clustering
eps = 1.0
min_samples = 2
dbscan_labels = vector_space.perform_dbscan(eps=eps, min_samples=min_samples)
print(
    f"DBSCAN clustering result with eps={eps} and min_samples={min_samples}: {dbscan_labels}"
)

# 可视化 3D
visualize_3d_vispy(vector_space)

# 序列化和反序列化
vector_space.to_json("vector_space.json")
restored_vector_space = VectorSpace.from_json("vector_space.json")
print("Restored VectorSpace from JSON:")
print(restored_vector_space)
print("----- end -----")
