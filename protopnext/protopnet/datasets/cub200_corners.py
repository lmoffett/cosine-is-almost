import logging
import os

import tqdm
from PIL import Image

log = logging.getLogger(__name__)


# adapted from
def crop_corners(base_dir, image_dir="train", output_dir="train_corner_crop"):
    """
    Subcommand to replicated prototree offline augmentation. If you plan
    on applying this method to a training set, make sure the seed from
    your split matches your validation and test set (view `create_splits`).

    Args:
        - base_dir: where your dataset is located (should include 'images.txt' and 'bounding_boxes.txt')
        - image_dir: the directory we want to perform augmentation on (should be the uncropped train)
        - output_dir: the name of output location (relative to the 'base_dir')
    """
    path = base_dir

    path_images = os.path.join(path, "images.txt")

    bbox_path = os.path.join(path, "bounding_boxes.txt")

    input_path = os.path.join(path, image_dir)
    output_path = os.path.join(path, output_dir)

    if os.path.isdir(output_path):
        log.error(f"Output directory already exists! Output directory: {output_path}")
        return

    full_image_name_to_id_map = dict()

    # open path to images
    with open(path_images, "r") as f:
        # get image index with file name
        for line in f:
            id, full_image_name = line.strip("\n").split(" ")
            full_image_name_to_id_map[full_image_name] = id

    bboxes = dict()
    with open(bbox_path, "r") as bf:
        for line in bf:
            line_list = line.split(" ")
            id = line_list[0]
            x, y, w, h = tuple(map(float, line_list[1:]))
            bboxes[id] = (x, y, w, h)

    log.info(f"Creating cropped corners datasets from {image_dir} to {output_dir}...")

    os.makedirs(output_path)
    for img_class in tqdm.tqdm(os.listdir(input_path)):
        os.makedirs(os.path.join(output_path, img_class))
        for img_name in os.listdir(os.path.join(input_path, img_class)):
            full_image_name = os.path.join(img_class, img_name)
            img = Image.open(os.path.join(input_path, full_image_name)).convert("RGB")

            (x, y, w, h) = bboxes[full_image_name_to_id_map[full_image_name]]
            width, height = img.size

            hmargin = int(0.1 * h)
            wmargin = int(0.1 * w)

            sample_name, ext = os.path.splitext(img_name)
            cropped_img = img.crop(
                (0, 0, min(x + w + wmargin, width), min(y + h + hmargin, height))
            )
            cropped_img.save(
                os.path.join(output_path, img_class, sample_name + "_upperleft" + ext)
            )
            cropped_img = img.crop(
                (0, max(y - hmargin, 0), min(x + w + wmargin, width), height)
            )
            cropped_img.save(
                os.path.join(output_path, img_class, sample_name + "_lowerleft" + ext)
            )
            cropped_img = img.crop(
                (max(x - wmargin, 0), 0, width, min(y + h + hmargin, height))
            )
            cropped_img.save(
                os.path.join(output_path, img_class, sample_name + "_upperright" + ext)
            )
            cropped_img = img.crop(
                ((max(x - wmargin, 0), max(y - hmargin, 0), width, height))
            )
            cropped_img.save(
                os.path.join(output_path, img_class, sample_name + "_lowerright" + ext)
            )

            img.save(os.path.join(output_path, img_class, sample_name + "_full" + ext))

    log.info(f"Corner crop complete! Directory: {output_path}")
