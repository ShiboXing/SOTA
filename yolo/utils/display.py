from PIL import Image, ImageDraw


def display_image_bbox(img, labels=[]):
    res_img = Image.fromarray((img * 255).permute(1, 2, 0).byte().numpy())
    draw = ImageDraw.Draw(res_img)
    for pc in labels:
        draw.rectangle(
            (448 * pc[0], 448 * pc[1], 448 * pc[2], 448 * pc[3]), outline="red"
        )
    res_img.show()
