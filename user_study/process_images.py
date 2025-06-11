import ast
import collections
import math
import os
import pathlib

import pandas as pd
import tqdm
from PIL import Image, ImageDraw, ImageFont

PATCH_HEIGHT = 212
QUERIES_PER_SEGMENT = 18
ATTENTION_PREFLIGHT = 3


def scale_bounding_box(x1, y1, x2, y2, W, H):

    scale_x = W / 224.0
    scale_y = H / 224.0

    left_x = x1 * scale_x
    upper_y = y1 * scale_y
    right_x = x2 * scale_x
    lower_y = y2 * scale_y

    return int(left_x), int(upper_y), int(right_x), int(lower_y)


def draw_box_on_image(image_path, bbox, match_x=False):
    """
    Draw a white box on an image at the specified coordinates.
    Returns the modified image and the cropped patch.
    """
    # Open the image
    cub200_path = pathlib.Path(os.environ["CUB200_DIR"])
    img = Image.open(cub200_path / image_path)
    draw = ImageDraw.Draw(img)

    # Convert string bbox to list if necessary
    if isinstance(bbox, str):
        bbox = ast.literal_eval(bbox)
    if os.environ.get("STUDY_SAMPLE_TYPE") != "attention":
        bbox = scale_bounding_box(*bbox, *img.size)
    part_left_x, part_upper_y, part_right_x, part_lower_y = bbox

    # Crop the patch
    patch = img.crop((part_left_x, part_upper_y, part_right_x, part_lower_y))

    desired_visual_thickness = PATCH_HEIGHT * 0.01
    future_img_height = int(PATCH_HEIGHT * 0.4)
    thickness_in_original = desired_visual_thickness * (img.size[1] / future_img_height)
    draw.rectangle(bbox, outline="red", width=max(1, int(round(thickness_in_original))))

    if match_x:
        # Match the width of the patch to the height
        new_x = min(int(img.size[0] * 0.8), patch.size[0] * 2)
        new_y = int(new_x * patch.size[1] / patch.size[0])
        enlarged_patch = patch.resize((new_x, new_y))
    else:
        # Match the height of the patch to the width
        new_y = min(int(img.size[1] * 0.8), patch.size[1] * 2)
        new_x = int(new_y * patch.size[0] / patch.size[1])
        enlarged_patch = patch.resize((new_x, new_y))

    return img, enlarged_patch


def create_reference_layout(ref_img, ref_patch):
    """Create the reference image layout with patch above."""
    # Calculate dimensions
    patch_width = int(PATCH_HEIGHT / ref_patch.size[1] * ref_patch.size[0])

    ref_patch = ref_patch.resize((patch_width, PATCH_HEIGHT))

    img_height = int(PATCH_HEIGHT * 0.4)
    img_width = int(img_height / ref_img.size[1] * ref_img.size[0])

    ref_img = ref_img.resize((img_width, img_height))

    total_width = max(patch_width, img_width, 200)
    total_height = PATCH_HEIGHT + img_height + 160

    # Create new image
    new_img = Image.new("RGB", (total_width, total_height), "white")

    # Calculate positions to center images
    patch_x = (total_width - ref_patch.size[0]) // 2
    img_x = (total_width - ref_img.size[0]) // 2

    # Paste images
    new_img.paste(ref_patch, (patch_x, 26))
    new_img.paste(ref_img, (img_x, ref_patch.size[1] + 164))

    # Create drawing object
    draw = ImageDraw.Draw(new_img)

    # Load a font (you can specify a different font file)
    try:
        font = ImageFont.truetype("arial.ttf", 18)  # Adjust size as needed
    except Exception:
        font = ImageFont.load_default(18)

    # Add text
    draw.text(
        (total_width // 2 - 70, ref_patch.size[1] + 140),
        "Reference Image",
        fill="black",
        font=font,
        align="center",
    )
    draw.text(
        (total_width // 2 - 70, 2),
        "Reference Patch",
        fill="black",
        font=font,
        align="center",
    )

    return new_img


def create_option_layout(main_img, patch, text=""):
    """Create option layout with image, patch, and text side by side."""
    # Calculate dimensions
    patch_width = int(PATCH_HEIGHT / patch.size[1] * patch.size[0])

    patch = patch.resize((patch_width, PATCH_HEIGHT))

    img_height = int(PATCH_HEIGHT * 0.65)
    img_width = int(img_height / main_img.size[1] * main_img.size[0])

    main_img = main_img.resize((img_width, img_height))

    # Add extra width for text
    total_width = max(main_img.size[0], 164) + max(patch.size[0], 100) + 120 + 4
    total_height = max(main_img.size[1], patch.size[1]) + 42

    # Create new image
    new_img = Image.new("RGB", (total_width, total_height), "#efefef")

    # Create drawing object
    draw = ImageDraw.Draw(new_img)

    # Load a font (you can specify a different font file)
    try:
        font = ImageFont.truetype("arial.ttf", 18)  # Adjust size as needed
    except Exception:
        font = ImageFont.load_default(18)

    y_offset_patch = (total_height - patch.size[1]) // 2 + 10
    y_offset_img = (total_height - main_img.size[1]) // 2 + 10

    x_offset = 4 if patch.size[0] > 88 else 44
    # Paste images
    new_img.paste(patch, (x_offset, y_offset_patch))
    new_img.paste(main_img, (patch.size[0] + 120, y_offset_img))

    # Add text
    draw.text(
        ((patch_width + x_offset) // 2 - 56 + x_offset // 2, 2),
        "Option Patch",
        fill="black",
        font=font,
        align="center",
    )
    draw.text(
        (patch_width + 120 + img_width // 2 - 56, 2),
        "Option Image",
        fill="black",
        font=font,
        align="center",
    )

    return new_img


prepped_sample = collections.namedtuple(
    "PreppedSamples", ["reference", "option_a", "option_b"]
)


def process_row(row):
    """Process a single row of the CSV data."""
    # Process reference image
    ref_img, ref_patch = draw_box_on_image(
        row["ref_path"], row["ref_box_mean"], match_x=True
    )
    ref_layout = create_reference_layout(ref_img, ref_patch)

    # Process option A
    opt_a_img, opt_a_patch = draw_box_on_image(
        row["proto_path_cos"], row["proto_box_cos"]
    )
    opt_a = create_option_layout(opt_a_img, opt_a_patch)

    # Process option B
    opt_b_img, opt_b_patch = draw_box_on_image(
        row["proto_path_l2"], row["proto_box_l2"]
    )
    opt_b = create_option_layout(opt_b_img, opt_b_patch)

    return prepped_sample(ref_layout, opt_a, opt_b)


def name_for_ith(i):
    if os.environ.get("STUDY_SAMPLE_TYPE") == "attention":
        if i < ATTENTION_PREFLIGHT:
            return f"s0_ac{i+1}"
        else:
            s = i + 1 - ATTENTION_PREFLIGHT
            return f"s{s}_ac"
    else:
        segment = int(math.floor((i / QUERIES_PER_SEGMENT))) + 1
        query = (i % QUERIES_PER_SEGMENT) + 1
        return f"s{segment}_q{query}"


def save_row(prepped_sample, i, output_dir):
    prefix = name_for_ith(i)
    ref_path = output_dir / f"{prefix}_ref.jpg"
    prepped_sample.reference.save(ref_path)
    option_a_path = output_dir / f"{prefix}_oa.jpg"
    prepped_sample.option_a.save(option_a_path)
    option_b_path = output_dir / f"{prefix}_ob.jpg"
    prepped_sample.option_b.save(option_b_path)
    return i, prefix, ref_path, option_a_path, option_b_path


def main(csv_path, output_dir: pathlib.Path):
    """Main function to process the CSV file."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV file
    df = pd.read_csv(csv_path).set_index("ref_id")

    output_df = pd.DataFrame(
        index=df.index,
        columns=[
            "i",
            "prefix",
            "ref_path",
            "cos_path",
            "l2_path",
        ],
    )

    # Process each row
    for i, (index, row) in tqdm.tqdm(enumerate(df.iterrows())):
        # Create a subdirectory for each row
        prepped_sample = process_row(row)
        row = save_row(prepped_sample, i, output_dir)
        output_df.loc[index] = row

    # Save the updated CSV
    if os.environ.get("STUDY_SAMPLE_TYPE") == "attention":
        output_path = output_dir / "attention_paths.csv"
    else:
        output_path = output_dir / "image_paths.csv"
    print("saving csv to", output_path)
    output_df.to_csv(output_path, index=True)


if __name__ == "__main__":
    study_csv_triplet_path = pathlib.Path(os.environ["STUDY_CSV_TRIPLET_PATH"])
    output_dir = pathlib.Path(os.environ["STUDY_OUTPUT_DIR"])

    print(f"Processing images from {study_csv_triplet_path} to {output_dir}")
    main(study_csv_triplet_path, output_dir)
