import numpy as np


def iou(bbox_A: np.ndarray, bbox_B: np.ndarray) -> float:
    """
    Method that computes the IoU metric (Intersection over Union)
    between two given bounding boxes.
    Both bounding boxes need to be in the form [x, y, width, height],
    where (x, y) are the coordinates of the top-left corner.

    It returns a float value in range [0, 1] inclusive.
    """

    # Extract parameters of the first bounding box
    Ax1, Ay1, Aw, Ah = bbox_A

    # Compute right-bottom corner coords of the first bounding box
    Ax2 = Ax1 + Aw
    Ay2 = Ay1 + Ah

    # Extract parameters of the second bounding box
    Bx1, By1, Bw, Bh = bbox_B

    # Compute right-bottom corner coords of the second bounding box
    Bx2 = Bx1 + Bw
    By2 = By1 + Bh

    # Compute the top-left and bottom-right coords of the intersection
    inter_x1 = max(Ax1, Bx1)
    inter_y1 = max(Ay1, By1)
    inter_x2 = min(Ax2, Bx2)
    inter_y2 = min(Ay2, By2)

    # Compute area of the intersection
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Compute each box' area
    areaA = Aw * Ah
    areaB = Bw * Bh

    # Compute area of the union
    union_area = areaA + areaB - inter_area

    # Handle the case of division by 0 in case of union area being equal to 0
    if union_area == 0:
        return 0.0

    return inter_area / union_area


def dist(bbox_A: np.ndarray, bbox_B: np.ndarray) -> float:
    """
    Method that computes the distance between the centers of
    two given bounding boxes.
    """

    # Extract parameters of both bounding boxes
    Ax, Ay, Aw, Ah = bbox_A
    Bx, By, Bw, Bh = bbox_B

    # Compute the coordinates of the bounding boxes' centers
    Ax_center = Ax + Aw / 2.0
    Ay_center = Ay + Ah / 2.0

    Bx_center = Bx + Bw / 2.0
    By_center = By + Bh / 2.0

    # Compute the distance between the bounding boxes' centers
    dist_x = Bx_center - Ax_center
    dist_y = By_center - Ay_center

    return np.sqrt(dist_x**2 + dist_y**2)
