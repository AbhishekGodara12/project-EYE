# Project EYE 

Project EYE is a computer vision-based application designed to [Identify eye state(open/close) in real time]. This project leverages state-of-the-art computer vision datasets and libraries to achieve high accuracy and performance.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Folder Structure](#folder-structure)
4. [Environment Setup](#environment-setup)
5. [Libraries and Dependencies](#libraries-and-dependencies)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

---

## Overview

Project EYE aims to [insert detailed description of the project]. It is built using Python and popular computer vision libraries such as OpenCV and Pytorch.

**ChatGpt 4-o was used to outline and organize code in final stage**

---

## Features

- [Feature 1: Eye State Detection]
- [Feature 2: Image preprocessing and augmentation]
- [Feature 3: Model training and evaluation]

---

## Folder Structure

```
project-EYE/
├── data/                                  # Main dataset directory
│   ├── test/                              # Contains test images for model evaluation
│   │   ├── close/                         # Test images of closed eyes
│   │   ├── open/                          # Test images of open eyes
│   ├── train/                             # Contains training images for model training
│       ├── close/                         # Training images of closed eyes
│       ├── open/                          # Training images of open eyes
│
├── local_image_eye_crops/                 # Folder for locally extracted eye region crops from webcam or sample images
│
├── local_images/                          # Raw local input images (captured or downloaded) before eye cropping
│
├── models/                                # Pretrained and fine-tuned model weights directory
│   ├── eye_detector_mobilenetv2_(MRL_dataset).pth    # Trained MobileNetV2 model on MRL Eye Dataset
│   ├── eye_detector_mobilenetv2_(OACE_dataset).pth   # Trained MobileNetV2 model on OACE Eye Dataset
│
├── main.ipynb                             # Main notebook for real-time eye detection and prediction demo
│
├── model_training(MRL).ipynb              # Notebook for training MobileNetV2 model using MRL Eye Dataset
│
├── model_training(OACE).ipynb             # Notebook for training MobileNetV2 model using OACE Eye Dataset
│
├── mrl_model_test.ipynb                   # Notebook for evaluating the MRL-trained model on test data
│
├── OACE_model_test.ipynb                  # Notebook for evaluating the OACE-trained model on test data
│
├── requirements.txt                       # List of Python dependencies (e.g., torch, torchvision, opencv-python, numpy)
│
├── README.md                              # Project documentation and usage instructions
│
└── LICENSE                                # License information for open-source usage and distribution
            
```

---

## Environment Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/project-EYE.git
    cd project-EYE
    ```
2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Libraries and Dependencies

The following libraries are used in this project:
- **OpenCV** (v4.5.5): For image processing.
- **Pytorch** (v2.5.1): For building and training machine learning models.
- **NumPy** (v1.23.0): For numerical computations.
- **Matplotlib** (v3.5.2): For data visualization.
- **scikit-learn** (v1.1.1): For additional machine learning utilities.

To install all dependencies, refer to the `requirements.txt` file.

---

## Usage

1. Prepare your dataset and place it in the `data/raw/` directory.
2. Run the preprocessing script:
    ```bash
    python src/preprocess.py
    ```
3. Train the model:
    ```bash
    python src/train.py
    ```
4. Test the application:
    ```bash
    python src/main.py
    ```

---

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

*Happy coding!* -BINOD