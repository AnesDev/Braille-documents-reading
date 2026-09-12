from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

def draw_labels(results_dir: Path):
    labels_dir = results_dir / "labels"
    predictions_dir = results_dir / "braille_characters"
    output_dir = results_dir / "labeled_docs"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except:
        font = ImageFont.load_default()

    def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
        return int((xc - w / 2) * img_w), int((yc - h / 2) * img_h), int((xc + w / 2) * img_w), int((yc + h / 2) * img_h)

    def cluster_and_sort(detections):
        detections = np.array(detections)
        detections = detections[detections[:,1].argsort()]
        heights = detections[:,3] - detections[:,1]
        avg_h = np.mean(heights) if len(heights) else 0
        row_break = np.diff(detections[:,1]) > (avg_h / 2 if avg_h>0 else 10)
        split_indices = np.where(row_break)[0] + 1
        rows = np.split(detections, split_indices)
        sorted_boxes = []
        for row in rows:
            row = row[row[:,0].argsort()]
            sorted_boxes.extend(row.tolist())
        return sorted_boxes

    for label_file in labels_dir.glob("*.txt"):
        stem = label_file.stem
        img_file = results_dir / f"{stem}.jpg"
        if not img_file.exists():
            img_file = results_dir / f"{stem}.png"
        if not img_file.exists():
            continue

        # load predictions and translated if available
        predictions_file = predictions_dir / stem / "predictions.txt"
        assembled_file = predictions_dir / stem / "assembled_braille.txt"
        translated_file = predictions_dir / stem / "translated.txt"

        if not predictions_file.exists():
            continue
        with open(predictions_file) as f:
            predictions = [line.strip() for line in f if line.strip()]

        assembled_text = assembled_file.read_text(encoding="utf-8") if assembled_file.exists() else ""
        translated_text = translated_file.read_text(encoding="utf-8") if translated_file.exists() else ""

        img = Image.open(img_file).convert("RGB")
        draw = ImageDraw.Draw(img)
        width, height = img.size

        detections = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                _, xc, yc, w, h = map(float, parts[:5])
                detections.append(yolo_to_xyxy(xc, yc, w, h, width, height))

        detections = cluster_and_sort(detections)
        for idx, (x1, y1, x2, y2) in enumerate(detections):
            if idx >= len(predictions): break
            # draw bbox label (predicted dot pattern or unicode)
            draw.text((x1, y2 + 2), predictions[idx], fill="red", font=font)

        # draw assembled braille and translated text at top-left area
        margin = 8
        # draw a semi-transparent rectangle for readability
        text_block = "Braille: " + assembled_text + "\nText: " + translated_text
        # naive wrapping: cut to fit width 
        draw.rectangle([0, 0, width, 60], fill=(255,255,255,200))
        draw.text((margin, margin), "Braille: " + assembled_text, fill="black", font=font)
        draw.text((margin, margin + 18), "Text: " + (translated_text if translated_text else "N/A"), fill="black", font=font)

        img.save(output_dir / f"{stem}_labeled.jpg")

    return output_dir
