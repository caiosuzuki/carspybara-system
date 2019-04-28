[1mdiff --git a/count_cars.py b/count_cars.py[m
[1mindex 94754a5..7dc0a25 100755[m
[1m--- a/count_cars.py[m
[1m+++ b/count_cars.py[m
[36m@@ -1,12 +1,14 @@[m
[31m-from darkflow.net.build import TFNet[m
[32m+[m[32mfrom darkflow.darkflow.net.build import TFNet[m
 import cv2[m
 import numpy as np[m
 [m
[31m-options = {"pbLoad": "built_graph/yolo-mor-4c.pb", "metaLoad": "built_graph/yolo-mor-4c.meta", "threshold": 0.3, "gpu": 0.5}[m
[32m+[m[32mINPUT_VIDEO_PATH = "/home/caio/repos/carspybara-system/input-videos"[m
[32m+[m
[32m+[m[32moptions = {"pbLoad": "built_graph/yolo-mor-4c.pb", "metaLoad": "built_graph/yolo-mor-4c.meta", "threshold": 0.3, "gpu": 0.8}[m
 [m
 tfnet = TFNet(options)[m
 [m
[31m-cap = cv2.VideoCapture("/home/takashi/exp/entradas/mor2.mp4")[m
[32m+[m[32mcap = cv2.VideoCapture(INPUT_VIDEO_PATH)[m
 counter = 0[m
 fps = cap.get(cv2.CAP_PROP_FPS)[m
 [m
