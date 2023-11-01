import torch
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.simple_cnn import SimpleCNN
from PIL import Image
from torchvision import transforms
import sys
import argparse 

from models.simple_cnn import SimpleCNN
from load_data import load_custom_dataset
from models.simple_cnn import SimpleCNN

def infer(video_path, output_filename, model_path):
    # creating an instance of our custom image classification model
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))  # Load model using the provided model path
    model.eval()

    # Define transformations for the inference image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(size=(50, 50)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))

    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    cv2.namedWindow("Video Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Inference", 800, 600)

    class_names = {0: 'crack', 1: 'good'}  # Define your class names

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for inference
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        input_tensor = transform(image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)

        probs = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

        predicted_label = class_names[predicted_class]
        text = f"Predicted Class: {predicted_label}"
        font_scale = 2
        font_color = (0, 0, 255)
        font_thickness = 2
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = frame.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

        cv2.imshow("Video Inference", frame)

        # Save the frame to the output video
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Inference on a video')
    parser.add_argument('--video_path', help='Path to the video file')
    parser.add_argument('--output_filename', help='Output filename for the annotated video')
    parser.add_argument('--model_path', help='Path to the model.pth file')
    parser.add_argument('--source', help='Specify the source (optional)')

    args = parser.parse_args()

    if args.source:  # If --source argument is provided, use its value for video_path
        video_path = args.source
    else:
        video_path = args.video_path

    # Call the infer function with the updated video_path
    infer(video_path, args.output_filename, args.model_path)
