# PyTorch-Image-Classifier 🖼️🚀
PyTorch Image Classifier: Train, validate, and classify images easily. Pre-trained models, custom model creation, data pipelines, and integration scripts included. Start building with PyTorch today! 🌟✨

# Image Classifier using PyTorch

Welcome to our Image Classifier repository! This project provides a complete solution for training, validating, and utilizing neural networks to classify images using PyTorch.

## File Structure

- `models/simple_cnn.py`: Contains the architecture for a simple CNN model. Contains a basic CNN architecture for image classification. We'll continually update with more CNN architectures, so stay tuned for updates!
- `infer.py`: Code for making inferences using the trained model. Infer real-time video using our model's weight file with the power of OpenCV and PyTorch.
- `load_data.py`: Script for loading and preprocessing data. 📂
- `loss.py`: Custom loss functions for model training. ❌
- `train.py`: Main script for training the image classifier model. 🏋️
- `train_iter.py`: Training loop iterations.

## Usage

- **Model Creation**: `models/simple_cnn.py` holds a basic CNN model structure. Modify or create your own model architecture here. 🧩
- **Training**: Use `train.py` to initiate model training. Customize with different datasets, hyperparameters, and more. 🚂
- **Inference**: `infer.py` provides a script to test and make predictions using the trained model. 🎥
- **Data Handling**: `load_data.py` is designed for data loading and preprocessing before training.

- **Usage**: Feel free to explore these files, utilize the provided scripts, and contribute to enhancing the image classification system.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Nivitus/PyTorch_Image_Classifier.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the Model**:
   ```bash
   python train.py --source path/path_of_the_dataset --dest path/where_you_wanna_save
   ```
4. **Make Predictions**:
   ```bash
   python infer.py --video_path /path/source_video.mp4 --output_filename /path/output_video.mp4 --model_path path/model_weights.pth
   ```

## Future Updates  🎉🔍📸
   We're dedicated to improving this repository. Stay tuned for more CNN architectures and updates to the training and inference functionalities!

## Happy Classifying! 🚀🔍📸
