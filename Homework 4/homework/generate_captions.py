import json
from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    from .generate_qa import extract_kart_objects, extract_track_info

    captions = []

    # Extract information
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track = extract_track_info(info_path)

    if not karts:
        return captions

    # Find ego car (center kart)
    ego_kart = next(k for k in karts if k["is_center_kart"])

    # 1. Ego car caption
    captions.append(f"{ego_kart['kart_name']} is the ego car.")

    # 2. Counting caption
    captions.append(f"There are {len(karts)} karts in the scenario.")

    # 3. Track name caption
    captions.append(f"The track is {track}.")

    # 4. Relative position captions for other karts
    for kart in karts:
        if kart["instance_id"] == ego_kart["instance_id"]:
            continue

        # Determine relative position
        if kart["center"][0] < ego_kart["center"][0]:
            left_right = "left"
        else:
            left_right = "right"

        if kart["center"][1] < ego_kart["center"][1]:
            front_behind = "in front"
        else:
            front_behind = "behind"

        captions.append(
            f"{kart['kart_name']} is {left_right} and {front_behind} of the ego car.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(
        f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(
        f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def generate_all_captions(data_dir: str, output_dir: str = "data/train"):
    """
    Generate caption pairs for all info files and save them as JSON files.

    Args:
        data_dir: Directory containing the info.json files
        output_dir: Directory to save the caption JSON files (default: "data/train")
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    info_files = list(data_path.rglob("*_info.json"))
    print(f"Found {len(info_files)} info.json files")

    for info_file in info_files:
        with open(info_file) as f:
            info = json.load(f)

        num_views = len(info.get("detections", []))
        frame_id_hex = info_file.stem.replace("_info", "")

        for view_index in range(num_views):
            try:
                captions = generate_caption(str(info_file), view_index)
                if not captions:
                    continue

                # Create caption pairs in the expected format
                image_filename = f"{frame_id_hex}_{view_index:02d}_im.jpg"
                caption_pairs = []

                for caption in captions:
                    caption_pairs.append({
                        "image_file": image_filename,
                        "caption": caption
                    })

                output_filename = f"{frame_id_hex}_{view_index:02d}_captions.json"
                output_path = out_path / output_filename

                with open(output_path, "w") as f_out:
                    json.dump(caption_pairs, f_out, indent=2)

                print(
                    f"Saved {output_filename} with {len(caption_pairs)} captions")

            except Exception as e:
                print(
                    f"Skipped {info_file} view {view_index} due to error: {e}")


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate_all_captions})


if __name__ == "__main__":
    main()
