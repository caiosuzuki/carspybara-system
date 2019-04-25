# import the necessary packages
import numpy as np

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh=0.35):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []


    # initialize the list of picked indexes
    pick = []

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    confidence = []

    for box in boxes:
        x1.append(box["topleft"]['x'])
        y1.append(box["topleft"]['y'])
        x2.append(box["bottomright"]['x'])
        y2.append(box["bottomright"]['y'])
        confidence.append(box["confidence"])


    # grab the coordinates of the bounding boxes
    x1 = np.asarray(x1).astype("float")
    y1 = np.asarray(y1).astype("float")
    x2 = np.asarray(x2).astype("float")
    y2 = np.asarray(y2).astype("float")

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(np.asarray(confidence))

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0]) ) )

    # return only the bounding boxes that were picked using the
    # integer data type
    return [boxes[i] for i in pick]
