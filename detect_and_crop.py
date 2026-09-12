from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import shutil
import numpy as np
import json

def detect_and_crop(model_path: Path, images_dir: Path, output_dir: Path):
    results_dir = output_dir / "results"
    labels_dir = results_dir / "labels"
    crops_dir = results_dir / "braille_characters"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    model.predict(
        source=str(images_dir),
        conf=0.1,
        iou=0.3,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(output_dir),
        name="results",
        exist_ok=True,
        show_labels=False,
        show_conf=False,
        line_width=1
    )

    def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
        x1 = max(int((xc - w / 2) * img_w), 0)
        y1 = max(int((yc - h / 2) * img_h), 0)
        x2 = min(int((xc + w / 2) * img_w), img_w)
        y2 = min(int((yc + h / 2) * img_h), img_h)
        return x1, y1, x2, y2

    def cluster_and_sort(detections):
        """
        detections: list of (x1,y1,x2,y2)
        returns: list of rows where each row is sorted left->right.
        Also we return a flattened list with row index and order preserved.
        """
        if not detections:
            return []

        arr = np.array(detections)
        # sort by top coordinate (y1)
        order_by_y = arr[arr[:,1].argsort()]
        heights = order_by_y[:,3] - order_by_y[:,1]
        avg_h = np.mean(heights) if len(heights) > 0 else 0
        # break rows where vertical gap between tops is larger than half avg height
        row_break = np.diff(order_by_y[:,1]) > (avg_h * 0.6 if avg_h > 0 else 10)
        split_indices = np.where(row_break)[0] + 1
        rows = np.split(order_by_y, split_indices)
        sorted_rows = []
        for r_idx, row in enumerate(rows):
            # sort row left -> right (by x1)
            row = row[row[:,0].argsort()]
            # convert to list of dicts with row index
            for box_idx, (x1, y1, x2, y2) in enumerate(row):
                sorted_rows.append({
                    "row": int(r_idx),
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "width": int(x2 - x1), "height": int(y2 - y1)
                })
        return sorted_rows

    # Crop characters and write metadata
    for label_file in labels_dir.glob("*.txt"):
        stem = label_file.stem
        img_file = next((images_dir / f"{stem}{ext}" for ext in [".jpg", ".png", ".jpeg"] if (images_dir / f"{stem}{ext}").exists()), None)
        if not img_file:
            # fallback: maybe YOLO saved the image in results_dir
            candidate = results_dir / f"{stem}.jpg"
            if candidate.exists():
                img_file = candidate
            else:
                candidate = results_dir / f"{stem}.png"
                if candidate.exists():
                    img_file = candidate

        if not img_file:
            print(f"Image not found for {label_file}")
            continue

        doc_folder = crops_dir / stem
        doc_folder.mkdir(parents=True, exist_ok=True)

        # open as RGB to avoid paletted-mode save errors
        img = Image.open(img_file).convert("RGB")
        width, height = img.size
        detections = []

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # YOLO label format: class xc yc w h
                # handle both 5-field and 6-field lines
                if len(parts) >= 6:
                    _, xc, yc, w, h, _ = parts[:6]
                else:
                    _, xc, yc, w, h = parts[:5]
                xc, yc, w, h = map(float, (xc, yc, w, h))
                x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, width, height)
                detections.append((x1, y1, x2, y2))

        sorted_meta = cluster_and_sort(detections)

        # save cropped images and metadata
        metadata = {"items": []}
        for idx, meta in enumerate(sorted_meta, start=1):
            x1, y1, x2, y2 = meta["x1"], meta["y1"], meta["x2"], meta["y2"]
            cropped = img.crop((x1, y1, x2, y2))
            # ensure crop is RGB before saving as JPEG
            if cropped.mode != "RGB":
                cropped = cropped.convert("RGB")
            fname = f"char_{idx}.jpg"
            cropped.save(doc_folder / fname)
            item = {
                "file": fname,
                "row": meta["row"],
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": meta["width"], "height": meta["height"],
                "order": idx
            }
            metadata["items"].append(item)

        # write metadata json
        with open(doc_folder / "metadata.json", "w", encoding="utf-8") as m:
            json.dump(metadata, m, indent=2, ensure_ascii=False)

    return results_dir, crops_dir
