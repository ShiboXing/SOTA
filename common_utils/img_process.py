import matplotlib.pyplot as plt
import matplotlib.patches as ptc


def show_bbox(img_pth, bbox: tuple):
    """
    show bounding box on an image
    @param img_pth: path to the image file
    @param bbox: (xmin, ymin, xmax, ymax) as absolute dimensions
    """
    img = plt.imread(img_pth)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    rect = ptc.Rectangle(
        bbox[:2],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)
    plt.imshow(img)
