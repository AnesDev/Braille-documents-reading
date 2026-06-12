# Braille Documents Reading

> An intelligent system for reading and converting Braille documents using computer vision and character recognition.

## 📋 Overview

The Braille Documents Reading project leverages computer vision and deep learning to automatically detect, recognize, and convert Braille characters from document images. This tool helps bridge accessibility gaps by enabling automated processing of Braille documents into readable text and vice versa.

## ✨ Features

- **📸 Braille Detection & Cropping**: Automatically detect and extract Braille text regions from document images
- **🔤 Character Recognition**: Use advanced ML models to recognize individual Braille characters with high accuracy
- **↔️ Text Conversion**: Convert between text and Braille formats
- **🎯 Label Drawing**: Visualize detected Braille characters with labeled output
- **🖥️ Web Interface**: User-friendly web application for easy interaction
- **📚 Multi-language Support**: Support for multiple languages and Braille standards

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AnesDev/Braille-documents-reading.git
   cd Braille-documents-reading
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies (if needed):**
   ```bash
   # On Ubuntu/Debian
   cat packages.txt | xargs apt-get install
   ```

## 📖 Usage

### Web Application

Launch the interactive web interface:

```bash
python app.py
```

The application will be available at `http://localhost:8000`

### Command Line Tools

#### Detect and Crop Braille from Images

```bash
python detect_and_crop.py input_image.jpg
```

#### Recognize Characters

```bash
python recognize_chars.py image_with_braille.jpg
```

#### Draw Labels on Detected Characters

```bash
python draw_labels.py detected_braille.jpg output_labeled.jpg
```

#### Text/Braille Conversion

```bash
python braille_utils.py
```

### Testing

Run the test suite:

```bash
python test.py
```

## 📁 Project Structure

```
Braille-documents-reading/
├── app.py                    # Main web application
├── recognize_chars.py        # Character recognition module
├── detect_and_crop.py        # Braille detection and cropping
├── draw_labels.py            # Label visualization
├── braille_utils.py          # Braille conversion utilities
├── test.py                   # Test suite
├── braille_transcriptor/     # Braille transcription module
├── models/                   # Pre-trained ML models
├── requirements.txt          # Python dependencies
└── packages.txt              # System dependencies
```

## 🛠️ Technical Stack

- **Python 3.7+**: Core language
- **Computer Vision**: Image processing and Braille detection
- **Deep Learning**: Character recognition with neural networks
- **Web Framework**: Interactive user interface

## 📦 Dependencies

Key packages include:
- OpenCV for image processing
- NumPy/SciPy for numerical computations
- TensorFlow/PyTorch for deep learning
- Flask/Streamlit for web interface

See `requirements.txt` for complete list.

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit** your changes:
   ```bash
   git commit -m "Add: description of your feature"
   ```
4. **Push** to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open** a Pull Request with a clear description of changes

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 🤔 FAQ

**Q: What image formats are supported?**  
A: JPG, PNG, and other common image formats supported by OpenCV.

**Q: How accurate is the character recognition?**  
A: Accuracy depends on image quality and Braille clarity. See test results for benchmarks.

**Q: Can it handle different Braille standards?**  
A: The system can be configured for various Braille standards and languages.

## 📧 Contact & Support

For questions or issues, please:
- Open an [GitHub Issue](https://github.com/AnesDev/Braille-documents-reading/issues)
- Check existing documentation

## 🙏 Acknowledgments

Thank you to all contributors and the open-source community for making accessibility technology possible.

---

**Last Updated:** March 2026
