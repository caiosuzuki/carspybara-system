from darkflow.net.build import TFNet
import numpy as np
import cv2
import sys
import copy
import json
import time
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.bbox_suppression import non_max_suppression_fast

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

def draw_bboxes(frame, bboxes, colors):
    for bbox in bboxes:
        confidence  = bbox["confidence"]
        topL = bbox["topleft"]
        bottomR = bbox["bottomright"]
        color = colors[ bbox["label"] ]

        cv2.rectangle(frame, (topL['x'], topL['y']), (bottomR['x'], bottomR['y']), color, 2)
        print_text_image(str(confidence)[:4], frame, (topL['x']+4, topL['y']+10), 0.3, color, 1)

def draw_centroid(frame, objs):
    for (objectID, centroid) in objs.items():
        cv2.circle(frame, (centroid[0], centroid[1]), 3, (0,0,255), -1)

def draw_line(img, line_points):
    color = (0, 255, 0)
    cv2.line(img, tuple(line_points[0]), tuple(line_points[1]), color, 5)

def draw_lines(img, lines):
    for line in lines:
        draw_line(img, line)

def print_text_image(string, image, bottom_right, size, color, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, string, bottom_right, font, size, color, thickness)

def draw_notification_board(image, ground_truth, estimated):

    h,w = image.shape[:2]

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



def get_frame_bboxes(tfnet, frame):
    return tfnet.return_predict(frame)

def load_lines(file_name):
    lines_file = open(file_name, "r")
    lines = json.loads(lines_file.read())['lines']
    lines_file.close()
    return lines

def get_lines_bboxes(lines):
    lines_bbxs = []

    for line in lines:
        topL_x, bottomR_x = (line[0][0], line[1][0]) if line[0][0] < line[1][0] else (line[1][0], line[0][0])
        topL_y, bottomR_y = (line[0][1], line[1][1]) if line[0][1] < line[1][1] else (line[1][1], line[0][1])

        bbox = {}
        bbox['topleft'] = {}
        bbox['bottomright'] = {}
        bbox['topleft']['x'] = topL_x
        bbox['topleft']['y'] = topL_y
        bbox['bottomright']['x'] = bottomR_x
        bbox['bottomright']['y'] = bottomR_y
        lines_bbxs.append(bbox)

    return lines_bbxs

def does_bbxs_intersect(a, b):
    c1 = a['topleft']['x'] < b['bottomright']['x']
    c2 = a['bottomright']['x'] > b['topleft']['x']
    c3 = a['topleft']['y'] < b['bottomright']['y']
    c4 = a['bottomright']['y'] > b['topleft']['y']

    intersect = c1 and c2 and c3 and c4
    return intersect

def filter_bboxes(bbxs, lines_bbxs):
    filtered_bboxes = []

    for line_bbx in lines_bbxs:
        filtered_bboxes.append([])
        for bbx in bbxs:
            if does_bbxs_intersect(bbx, line_bbx):
                filtered_bboxes[-1].append(bbx)

    return filtered_bboxes

def convert_bboxes_format(bboxes):
    new_format = []

    for bbox in bboxes:
        topL = bbox['topleft']
        bottomR = bbox['bottomright']
        new_format.append(np.array([topL['x'], topL['y'], bottomR['x'], bottomR['y']]))

    return new_format

def separate_bbox_by_class(bboxes, classes):
    separated = []
    for i in range(len(classes)):
        separated.append([])

    for bbox in bboxes:
        index = classes.index(bbox["label"])
        separated[index].append(bbox)

    return separated

def compute_video(cap, lines, video_writer, ground_truth):

    classes = ["car", "bus", "truck", "motorcycle"]
    lines_bbxs = get_lines_bboxes(lines)

    trackers = [CentroidTracker(maxDisappeared=10) for _ in range(len(lines_bbxs))]
    fps = 60
    ret = True
    t = 0
    while cap.isOpened() and ret:
        ret, frame = cap.read()
        if ret:
            bboxes = get_frame_bboxes(tfnet, frame)

        draw_lines(frame, lines)

        intersectig_bboxes = filter_bboxes(bboxes, lines_bbxs)
        total = 0
        for i,intersection in enumerate(intersectig_bboxes):

            # separated = separate_bbox_by_class(intersection, classes)

            boxes = non_max_suppression_fast(intersection)
            # for k in range(len(separated)):
            #     # boxes += non_max_suppression_fast(separated[k])

            objs = trackers[i].update(convert_bboxes_format(boxes))
            total += trackers[i].nextObjectID

            draw_bboxes(frame, boxes, colors)
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

    compute_video(cap, lines, video_writer, int(sys.argv[6]))

    cap.release()
    video_writer.release()
