# visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from vispy import scene
from vispy.scene import visuals
from collections import defaultdict


def visualize_2d(vector_space, dimension_indices=None):
    """
    Visualize the points in 2D space using the specified dimensions.
    Parameters:
    vector_space (VectorSpace): The VectorSpace instance to visualize.
    dimension_indices (list of int): Indices of the dimensions to visualize. Default is the first two dimensions.
    """
    if dimension_indices is None:
        dimension_indices = [0, 1]
    if len(dimension_indices) != 2:
        raise ValueError("Two dimensions must be specified for 2D visualization.")
    if any(index >= len(vector_space.dimensions) for index in dimension_indices):
        raise ValueError("Dimension index out of range.")
    fig, ax = plt.subplots()
    # 绘制原点
    ax.scatter(0, 0, color="blue", s=100)  # 原点用蓝色大点表示
    ax.text(0, 0, "Origin", color="blue", fontsize=12, ha="right")
    for point in vector_space.points.values():
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
    ax.set_xlabel(vector_space.dimensions[dimension_indices[0]].name)
    ax.set_ylabel(vector_space.dimensions[dimension_indices[1]].name)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()


def visualize_3d(vector_space, dimension_indices=None):
    """
    Visualize the points in 3D space using the specified dimensions.
    Parameters:
    vector_space (VectorSpace): The VectorSpace instance to visualize.
    dimension_indices (list of int): Indices of the dimensions to visualize. Default is the first three dimensions.
    """
    if dimension_indices is None:
        dimension_indices = [0, 1, 2]
    if len(dimension_indices) != 3:
        raise ValueError("Three dimensions must be specified for 3D visualization.")
    if any(index >= len(vector_space.dimensions) for index in dimension_indices):
        raise ValueError("Dimension index out of range.")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # 绘制原点
    ax.scatter(0, 0, 0, color="blue", s=100, label="Origin")  # 原点用蓝色大点表示
    ax.text(0, 0, 0, "Origin", color="blue", fontsize=12, ha="right")
    for point in vector_space.points.values():
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
    ax.set_xlabel(vector_space.dimensions[dimension_indices[0]].name)
    ax.set_ylabel(vector_space.dimensions[dimension_indices[1]].name)
    ax.set_zlabel(vector_space.dimensions[dimension_indices[2]].name)
    plt.legend()
    plt.show()


def visualize_3d_vispy(
    vector_space, dimension_indices=None, cluster_key="kmeans_clusters"
):
    if dimension_indices is None:
        dimension_indices = [0, 1, 2]
    if len(dimension_indices) != 3:
        raise ValueError("Three dimensions must be specified for 3D visualization.")
    if any(index >= len(vector_space.dimensions) for index in dimension_indices):
        raise ValueError("Dimension index out of range.")

    # 创建一个场景画布
    canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="white")
    view = canvas.central_widget.add_view()
    view.camera = "turntable"  # 设置相机类型

    # 创建散点并赋予颜色
    scatter = visuals.Markers()
    point_lst = list(vector_space.points.values())
    points = np.array([point.coordinates for point in point_lst])

    # 计算质心
    centroid = points[:, dimension_indices].mean(axis=0)
    view.camera.center = centroid  # 设置相机旋转中心为质心

    # 生成颜色字典，每个类一个颜色
    color_map = defaultdict(lambda: np.random.rand(3))

    # 生成每个点的颜色
    colors = np.array(
        [
            color_map[point.labels.get(cluster_key, np.random.randint(0, 10000))]
            for point in point_lst
        ]
    )

    scatter.set_data(
        points[:, dimension_indices], edge_color=colors, face_color=colors, size=16
    )
    view.add(scatter)

    # 添加坐标轴
    axis = visuals.XYZAxis(parent=view.scene)
    # 添加标尺
    grid = visuals.GridLines()
    view.add(grid)

    # 显示原点
    origin = visuals.Markers()
    origin.set_data(np.array([[0, 0, 0]]), face_color="blue", size=18)
    view.add(origin)

    # 为坐标轴添加标签
    axis_labels = {
        vector_space.dimensions[dimension_indices[0]].name: (1, 0, 0),
        vector_space.dimensions[dimension_indices[1]].name: (0, 1, 0),
        vector_space.dimensions[dimension_indices[2]].name: (0, 0, 1),
    }
    for label, pos in axis_labels.items():
        text = visuals.Text(
            text=label,
            color="black",
            anchor_x="center",
            anchor_y="center",
            font_size=36,
            bold=True,
        )
        text.pos = np.array(pos) * 1.1  # 让标签稍微远离原点
        view.add(text)

    # 在每个点旁边显示文本标签并稍微向上移动
    for i, point in enumerate(points[:, dimension_indices]):
        point_label = visuals.Text(
            text=f"{point_lst[i].id} - {point_lst[i].name}",
            color=colors[i],
            anchor_x="center",
            anchor_y="center",
            font_size=8,
            bold=True,
        )
        point_label.pos = point + np.array([0, 0.01, 0])  # 向上移动一点
        view.add(point_label)

    canvas.app.run()


if __name__ == "__main__":
    print("visualization start")
