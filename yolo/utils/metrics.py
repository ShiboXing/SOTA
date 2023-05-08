
def xywh_2_xxyy(bbox):
    """Bbox: x, y, w, h"""
    return (
        bbox[0] - bbox[2] / 2,
        bbox[1] - bbox[3] / 2,
        bbox[0] + bbox[2] / 2,
        bbox[1] + bbox[3] / 2,
    )

def IOU(output, label):
    """Calculate the intersection over union of two sets rectangles

    Keyword arguments
    both output and label are (x, y, w, h)
    """
    
    output = xywh_2_xxyy(output)
    label = xywh_2_xxyy(label)

    x_inter = min(output[2], label[2]) - max(output[0], label[0])
    y_inter = min(output[3], label[3]) - max(output[1], label[1])

    if x_inter <= 0.0 or y_inter <= 0.0:
        return 0.0

    intersection = x_inter * y_inter

    overlapped_union = (output[2] - output[0]) * (output[3] - output[1]) + (label[2] - label[0]) * (
        label[3] - label[1]
    )
    
    return intersection / (overlapped_union - intersection)

# sanity checks of IOU
coords = (0.25, 0.25, 0.5, 0.5)
y_coords1 = (0.5, 0.575, 0.5, 0.35)
y_coords2 = (0.575, 0.5, 0.35, 0.5)
y_coords3 = (0.25, 0.25, 0.5, 0.5)
y_coords4 = (0.25, 0, 0.5, 0)
y_coords5 = (0.625, 0.375, 0.25, 0.75)
y_coords6 = (0.2, 0.25, 0.2, 0.3)

def float_eqs(a, b, decimal_pt):
    eps = 10 ** (-decimal_pt)
    return abs(a-b) < eps

assert float_eqs(IOU(coords, y_coords1), 0.025 / (0.5*0.5 + 0.5*0.35 - 0.025), 5)
assert float_eqs(IOU(coords, y_coords2), 0.025 / (0.5*0.5 + 0.5*0.35 - 0.025), 5)
assert float_eqs(IOU(coords, y_coords3), 1, 5)
assert float_eqs(IOU(coords, y_coords4), 0, 5)
assert float_eqs(IOU(coords, y_coords5), 0, 5)
assert float_eqs(IOU(coords, y_coords6), 0.06 / (0.5*0.5), 5)