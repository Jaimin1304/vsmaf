import numpy as np
from vispy import scene
from vispy.scene import visuals

# 创建一个场景画布
canvas = scene.SceneCanvas(keys="interactive", show=True, bgcolor="white")
view = canvas.central_widget.add_view()
view.camera = "turntable"  # 设置相机类型

# 创建一些3D点数据
points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
labels = ["A", "B", "C", "D"]

# 创建散点
scatter = visuals.Markers()
scatter.set_data(points, face_color="red", size=10)
view.add(scatter)

# 在每个点上显示名称
for point, label in zip(points, labels):
    text = visuals.Text(
        text=label, color="black", anchor_x="left", anchor_y="bottom", font_size=12
    )
    text.pos = point
    view.add(text)

# 添加坐标轴
axis = visuals.XYZAxis(parent=view.scene)

# 显示原点
origin = visuals.Markers()
origin.set_data(np.array([[0, 0, 0]]), face_color="blue", size=10)
view.add(origin)

# 为坐标轴添加标签
axis_labels = {
    "X": (1, 0, 0),
    "Y": (0, 1, 0),
    "Z": (0, 0, 1),
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
