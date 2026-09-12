import streamlit as st
from pathlib import Path
import tempfile
import shutil
from PIL import Image

from detect_and_crop import detect_and_crop
# updated recognize_chars signature
from recognize_chars import recognize_characters
from draw_labels import draw_labels

# CONFIG
st.set_page_config(page_title="Braille Detection", layout="centered")

# MODELS PATHS
YOLO_MODEL_PATH = Path("models") / "yolo8l.pt"
RECOG_MODEL_PATH = Path("models") / "recognition_model.pth"

st.title("Braille Document Recognition")

# language + grade selector - ADDED RUSSIAN
language = st.selectbox("Document language", ["English", "French", "Arabic", "Russian"])
# grade_choice = st.selectbox("Braille Grade", ["Grade 1", "Grade 2"])
# grade = 1 if grade_choice == "Grade 1" else 2
grade = 1
space_factor = st.slider("Word gap sensitivity", 1.0, 2.0, 1.2, 0.05)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Prepare directories
        images_dir = temp_dir / "input"
        images_dir.mkdir(parents=True, exist_ok=True)
        output_dir = temp_dir / "output"

        # Save uploaded file
        input_path = images_dir / uploaded_file.name
        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        st.info("Running detection and cropping...")
        results_dir, crops_dir = detect_and_crop(YOLO_MODEL_PATH, images_dir, output_dir)

        st.info("Recognizing Braille characters...")
        recognize_characters(crops_dir, RECOG_MODEL_PATH, language=language, grade=grade, space_factor=space_factor)

        # ADD THIS: Show the assembled Braille before translation
        assembled_braille_file = results_dir / "braille_characters" / input_path.stem / "assembled_braille.txt"
        if assembled_braille_file.exists():
            assembled_braille = assembled_braille_file.read_text(encoding="utf-8")
            st.subheader("Assembled Braille (before translation)")
            st.text_area("Raw Braille Unicode", assembled_braille, height=100)

        st.info("Drawing labels on the document...")
        labeled_docs_dir = draw_labels(results_dir)

        # Show final labeled image
        labeled_images = list(labeled_docs_dir.glob("*_labeled.jpg"))
        if labeled_images:
            labeled_image_path = labeled_images[0]
            labeled_image = Image.open(labeled_image_path)
            st.image(labeled_image, caption="Labeled Braille Document", use_container_width=True)

            # show translated text file contents if present
            translated_file = (results_dir / "braille_characters" / labeled_image_path.stem.replace("_labeled","") / "translated.txt")
            if translated_file.exists():
                translated_text = translated_file.read_text(encoding="utf-8")
                st.subheader("Translated Text")
                st.text_area("Translation", translated_text, height=200, disabled=True)

            # Download
            with open(labeled_image_path, "rb") as f:
                st.download_button(
                    label="Download Labeled Image",
                    data=f,
                    file_name=labeled_image_path.name,
                    mime="image/jpeg"
                )
        else:
            st.error("No labeled image produced.")
