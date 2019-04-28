from typing import List, Dict, Tuple
from geometry import Vec2, BoundingBox
from thtrieu_darkflow.darkflow.net.build import TFNet
import numpy as np
import cv2
import sys
import copy
import json
import time
from pyimagesearch.boundingboxtracker import BoundingBoxTracker
from pyimagesearch.bbox_suppression import non_max_suppression_fast
from utils import Prediction

def draw_line(img, line_points):
    color = (0, 255, 0)
    cv2.line(img, tuple(line_points[0]), tuple(line_points[1]), color, 5)

def draw_point(img, point):
    color = (0,0,255)
    cv2.circle(img, tuple(point), 2, color, -1)

def draw_components(img, lines):
    for line in lines:
        if len(line) == 2:
            draw_line(img, line)

        for point in line:
            draw_point(img, point)

def draw_bboxes(frame, predictions: List[Prediction], colors: dict) -> None:
    for p in predictions:
        confidence  = p.confidence
        tl = p.bbox.top_left
        br = p.bbox.bottom_right
        color = colors[p.label]
        cv2.rectangle(frame, (tl.x, tl.y), (br.x, br.y), color, 2)
        print_text_image(str(confidence)[:4], frame, (tl.x, tl.y+10), 0.3, color, 1)

def draw_centroid(frame: np.ndarray, objs) -> None:
    for bbox in objs.values():
        centroid = (bbox.top_left + bbox.bottom_right) / 2
        cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 2, (0,0,255), -1)

def draw_lines(img, lines):
    for line in lines:
        draw_line(img, line)

def print_text_image(string, image, bottom_right, size, color, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, string, bottom_right, font, size, color, thickness)

def draw_notification_board(image, ground_truth, estimated):

    _,w = image.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    str1 = "GROUND TRUTH: %4d" % ground_truth
    str2 = "ESTIMATED     : %4d" % estimated

    str_size = cv2.getTextSize(str1, font, scale, thickness)
    str_width = str_size[0][0]
    str_height = str_size[0][1]

    rect_width = str_width + 20
    rect_height = 2*str_height + 20

    # draw rect
    cv2.rectangle(image,(w-rect_width-1, 0), (w-1, rect_height), (33,33,33), -1)
    print_text_image(str1, image, (w-str_width-10, str_height), scale, (146,189,103), thickness)
    print_text_image(str2, image, (w-str_width-10, 2*str_height+10), scale, (81,52,216), thickness)

def get_frame_predictions(tfnet: TFNet, frame: np.ndarray) -> List[Prediction]:
    net_predictions = tfnet.return_predict(frame)

    predictions = []
    for p in net_predictions:
        tl = Vec2(p['topleft']['x'], p['topleft']['y'])
        br = Vec2(p['bottomright']['x'], p['bottomright']['y'])
        label = p['label']
        confidence = p['confidence']
        predictions.append(Prediction(BoundingBox(tl, br), label, confidence))
    
    return predictions

def load_lines(file_name: str) -> List:
    with open(file_name, "r") as lines_file:
        lines = json.loads(lines_file.read())['lines']
    return lines

def get_lines_bboxes(lines: List[List[int]]) -> List[BoundingBox]:
    lines_bbxs = []

    for line in lines:
        topL_x, bottomR_x = (line[0][0], line[1][0]) \
                    if line[0][0] < line[1][0] else (line[1][0], line[0][0])
        topL_y, bottomR_y = (line[0][1], line[1][1]) \
                    if line[0][1] < line[1][1] else (line[1][1], line[0][1])
        lines_bbxs.append(BoundingBox(Vec2(topL_x, topL_y), Vec2(bottomR_x, bottomR_y)))

    return lines_bbxs

def filter_predictions(predictions: List[Prediction],
                       lines_bbxs: List[BoundingBox]) -> List[List[Prediction]]:
    
    index = set(i for i in range(len(predictions)))

    filtered_predictions = []
    for line_bbx in lines_bbxs:
        filtered_predictions.append([])
        b = []

        for k in index:
            if predictions[k].bbox.intersect(line_bbx):
                filtered_predictions[-1].append(predictions[k])
                b.append(k)

        index -= set(b)

    return filtered_predictions

def compute_video(cap, lines: List, 
                  video_writer: cv2.VideoWriter,
                  ground_truth: int,
                  colors: Dict[str, Tuple]) -> None:
    
    lines_bbxs = get_lines_bboxes(lines)
    trackers = [BoundingBoxTracker(maxDisappeared=10) for _ in range(len(lines_bbxs))]
    fps = 60
    ret = True
    t = 0
    while cap.isOpened() and ret:
        ret, frame = cap.read()
        if ret:
            predictions = get_frame_predictions(tfnet, frame)

        draw_lines(frame, lines)

        filter_pred = filter_predictions(predictions, lines_bbxs)
        total = 0

        for i, pred_set in enumerate(filter_pred):
            indexes = non_max_suppression_fast(pred_set)
            final_pred = [pred_set[k] for k in indexes]
            objs = trackers[i].update([p.bbox for p in final_pred])
            total += trackers[i].nextObjectID

            draw_bboxes(frame, final_pred, colors)
            draw_centroid(frame, objs)

        draw_notification_board(frame, ground_truth, total)

        video_writer.write(frame)
        t += 1
        if t >= fps:
            print("total, ", total)
            t = 0

if __name__ == "__main__":
    colors = {}
    colors['motorcycle'] = (255, 226, 5)
    colors['car'] = (186, 82, 15)
    colors['truck'] = (6, 37, 82)
    colors['bus'] = (2, 106, 253)

    options = {"pbLoad": sys.argv[1], "metaLoad": sys.argv[2], "threshold": 0.3, "gpu": 1.0}
    tfnet = TFNet(options)

    cap = cv2.VideoCapture(sys.argv[3])
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_writer = cv2.VideoWriter(sys.argv[4], fourcc, fps, (width, height))
    lines = load_lines(sys.argv[5])

    compute_video(cap, lines, video_writer, int(sys.argv[6]), colors)

    cap.release()
    video_writer.release()
