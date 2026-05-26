import ultralytics.nn.tasks as tasks
from fruit_detection.cbam import CBAM

# 注入到 YAML 解析器所在模块的全局命名空间
tasks.__dict__["CBAM"] = CBAM

from ultralytics import YOLO

model = YOLO("configs/yolov8n-cbam.yaml")
print("✅ CBAM 模型构建成功")
print(model.model)