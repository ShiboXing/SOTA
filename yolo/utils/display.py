from PIL import Image, ImageDraw, ImageFont


def display_image_bbox(img, labels=[], titles=[]):
    res_img = Image.fromarray((img * 255).permute(1, 2, 0).byte().numpy())
    draw = ImageDraw.Draw(res_img)
    for i, pc in enumerate(labels):
        draw.rectangle(
            (448 * pc[0], 448 * pc[1], 448 * pc[2], 448 * pc[3]), outline="red"
        )

        text = titles[i]
        font = ImageFont.truetype("arial.ttf", 40)
        # text_width, text_height = draw.textsize(text)
        # x = (width - text_width) // 2
        # y = (height - text_height) // 2
        draw.text(
            (
                448 * (pc[0] + pc[2] / 2 - pc[0] / 2),
                448 * (pc[1] + pc[3] / 2 - pc[1] / 2),
            ),
            text,
            fill="red",
            font=font,
        )
    res_img.show()
