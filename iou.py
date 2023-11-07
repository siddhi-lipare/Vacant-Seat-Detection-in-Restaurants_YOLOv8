def calculate_iou(box1, box2):
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

    # Calculate the intersection coordinates
    x_left = max(x1_box1, x1_box2)
    y_top = max(y1_box1, y1_box2)
    x_right = min(x2_box1, x2_box2)
    y_bottom = min(y2_box1, y2_box2)

    if x_right > x_left and y_bottom > y_top:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
        box2_area = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)
        iou = intersection_area / (box1_area + box2_area - intersection_area)
        return iou
    else:
        return 0