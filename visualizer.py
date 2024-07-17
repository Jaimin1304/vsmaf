# visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from vispy import scene
from vispy.scene import visuals


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


def visualize_3d_vispy(vector_space, dimension_indices=None):
    """
    Visualize the points in 3D space using the specified dimensions with GPU acceleration.
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
    # 创建一个场景画布
    canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="white")
    view = canvas.central_widget.add_view()
    view.camera = "turntable"  # 设置相机类型
    # 创建散点并赋予随机颜色
    scatter = visuals.Markers()
    points = np.array([point.coordinates for point in vector_space.points.values()])
    colors = np.random.rand(len(points), 3)  # 生成随机颜色
    scatter.set_data(
        points[:, dimension_indices], edge_color=colors, face_color=colors, size=8
    )
    view.add(scatter)
    # 添加坐标轴
    axis = visuals.XYZAxis(parent=view.scene)
    # 添加标尺
    grid = visuals.GridLines()
    view.add(grid)
    # 显示原点
    origin = visuals.Markers()
    origin.set_data(np.array([[0, 0, 0]]), face_color="blue", size=10)
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
            font_size=20,
            bold=True,
        )
        text.pos = np.array(pos) * 1.1  # 让标签稍微远离原点
        view.add(text)

    # 添加鼠标悬停事件处理器
    def on_mouse_move(event):
        if event.pos is None:
            return
        tr = scatter.get_transform("canvas", "visual")
        mouse_pos = tr.map(event.pos)[:3]
        min_dist = float("inf")
        nearest_point = None
        for i, point in enumerate(points[:, dimension_indices]):
            dist = np.linalg.norm(mouse_pos - point)
            if dist < min_dist:
                min_dist = dist
                nearest_point = i
        if nearest_point is not None:
            point_coords = points[nearest_point, dimension_indices]
            print(f"Mouse over: {point_coords}")

    canvas.events.mouse_move.connect(on_mouse_move)
    canvas.app.run()


if __name__ == "__main__":
    print("visualization start")
