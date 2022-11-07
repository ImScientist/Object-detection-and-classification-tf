import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from data.preprocessing import mask_from_compact_notation


def cloud_colors():
    """ Colors assigned to each cloud type """

    colors = {
        'Sugar': (128, 0, 0, 255),
        'Gravel': (0, 128, 0, 255),
        'Flower': (0, 0, 128, 255),
        'Fish': (128, 0, 128, 255)}

    return colors


def visualize_colors_map():
    """ Segments colr map """

    colors = cloud_colors()

    img = Image.new(mode='RGBA', size=(240, 30), color=(0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    for idx, (k, v) in enumerate(colors.items()):
        draw.text(xy=(10 + 60 * idx, 10), text=k, fill=v)

    plt.figure(figsize=(8, 3))
    plt.imshow(img)
    plt.pause(0.001)
    plt.show()


def visualize_labeled_image(
        img_path: str,
        data: dict[str, list[int]],
        downscale: int = 4,
        img_height: int = 1400,
        img_width: int = 2100
):
    """ Create an image with the regions of different cloud types

    Parameters
    ----------
      img_path:
      data: cloud type (key) to compact mask definition (value)
      downscale: how many times to downscale the image
      img_height:
      img_width:
    """

    colors = cloud_colors()

    masks = {k: mask_from_compact_notation(v, img_height, img_width)
             for k, v in data.items()}

    img = Image.open(img_path).convert('RGBA')
    img.putalpha(256)

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    drawing = ImageDraw.Draw(overlay)

    for k, mask in masks.items():
        new_mask = Image.fromarray((mask * 110).astype(np.uint8), mode='L')
        drawing.bitmap((0, 0), new_mask, fill=colors[k])

    img = Image.alpha_composite(img, overlay)
    img = img.resize((img.width // downscale, img.height // downscale))

    return img


def visualize_multiple_predictions(y, y_hat, n: int = 5):
    """ Visualize predicted segments for multiple images

    Parameters
    ----------
      y: tensor or np.array with shape(None, widht, height, 4)
      y_hat: tensor or np-array with shape(None, widht, height, 4)
      n: number of images
    """

    assert y.ndim == 4
    assert y_hat.ndim == 4

    figsize = (12, 4 * n)
    fig = plt.figure(figsize=figsize)

    idx = 1
    for r in range(n):
        for c in range(4):
            plt.subplot(2 * n, 4, idx)
            plt.title('segments')
            plt.imshow(y[r, ..., c])
            plt.axis('off')

            plt.subplot(2 * n, 4, idx + 4)
            plt.title('predictions')
            plt.imshow(y_hat[r, ..., c])
            plt.axis('off')

            idx += 1
        idx += 4

    plt.tight_layout()
    return fig


def visualize_image_augmentations(x, x_aug, y, y_aug, figsize=(22, 7)):
    """ Visualize image amd segment augmentations

    Parameters
    ----------
      x: image
      x_aug: augmented image
      y: segments
      y_aug: augmented segments
      figsize:
    """

    fig = plt.figure(figsize=figsize)

    plt.subplot(2, 5, 1)
    plt.title('Image')
    plt.imshow(x)

    plt.subplot(2, 5, 6)
    plt.title('Augmented image')
    plt.imshow(x_aug)

    for i in range(4):
        plt.subplot(2, 5, 2 + i)
        plt.title('Segments')
        plt.imshow(y[..., i])

        plt.subplot(2, 5, 7 + i)
        plt.title('Augmented segments')
        plt.imshow(y_aug[..., i])

    plt.tight_layout()

    return fig
