import cv2
from ultralytics import YOLO
import os
from datetime import datetime

# 1. 加载模型（请确保 best.pt 放在项目文件夹里）
model = YOLO("yolov8n_fruits_best.pt")

# 2. 打开摄像头（0 表示默认摄像头）
cap = cv2.VideoCapture(0)

# 3. 设置画面大小（让检测更快）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 4. 创建一个用来保存截图的文件夹（如果还没有的话）
save_folder = "截图保存"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"已创建文件夹：{save_folder}")

print("摄像头已打开，按 Q 键退出，按 S 键保存当前画面")

while True:
    success, frame = cap.read()
    if not success:
        print("读取摄像头失败，请检查摄像头是否被其他软件占用")
        break

    # 5. 用模型检测这一帧
    results = model(frame, verbose=False)

    # 6. 在画面上画出识别框和水果名字
    annotated_frame = results[0].plot()

    # 7. 显示画面
    cv2.imshow("水果实时检测 (Q退出 S保存)", annotated_frame)

    # 8. 检查按键
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):          # 按 Q 退出
        break
    elif key == ord('s'):        # 按 S 保存当前画面
        # 生成带时间的文件名，例如：截图保存/20260522_153022.jpg
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_folder, f"fruit_{timestamp}.jpg")
        cv2.imwrite(filename, annotated_frame)
        print(f"已保存：{filename}")

# 9. 关闭摄像头和窗口
cap.release()
cv2.destroyAllWindows()
print("已退出")