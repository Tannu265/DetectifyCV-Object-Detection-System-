# 🎯 DetectifyCV
Real-Time Object Detection using OpenCV & Deep Learning
---

# 📖 Overview

DetectifyCV is a real-time object detection system built with Python and OpenCV's Deep Neural Network (DNN) module.
It leverages the pre-trained MobileNet SSD (Single Shot MultiBox Detector) model to detect and classify objects directly from a live webcam feed.
It performs real-time object detection using a lightweight deep learning model optimized for speed and efficiency. The system captures live video frames, processes them through a pre-trained neural network, and displays detected objects with bounding boxes and confidence scores.
---

# 🚀 Features

* 🎥 Real-time webcam detection

* 🧠 Deep learning-based object recognition

* 📦 Bounding box visualization

* 🏷️ Class labels with confidence scores

* ⚡ Lightweight and fast execution

* 💻 Command-line interface support
---

# 🛠️ Tech Stack

*Language: Python 3.x
*Computer Vision: OpenCV
*Deep Learning Model: MobileNet SSD (Caffe framework)
*Libraries: NumPy, imutils
---

# 📂 Project Structure
DetectifyCV/
│
├── real_time_object_detection.py
├── MobileNetSSD_deploy.prototxt.txt
├── MobileNetSSD_deploy.caffemodel
├── requirements.txt
└── README.md
---

## ⚙️ Installation
# 1️⃣ Clone the Repository
git clone https://github.com/Tannu265/DetectifyCV-Object-Detection-System-.git
cd DetectifyCV

# 2️⃣ Create Virtual Environment (Recommended)

Windows:

python -m venv venv
venv\Scripts\activate

# 3️⃣ Install Dependencies
pip install -r requirements.txt

Or manually:

pip install opencv-python numpy imutils

# ▶️ Usage

Run the following command from the project directory:

python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel


The webcam will launch automatically and begin detecting objects in real time.

Press q to exit.
---

# 🧠 How It Works

Loads the pre-trained MobileNet SSD model using OpenCV’s DNN module.

Captures live frames from the webcam.

Converts frames into a blob format for neural network processing.

Runs forward pass through the network.

Extracts detection results above a confidence threshold.

Draws bounding boxes and labels on detected objects.

📊 Model Information

Model: MobileNet SSD

Framework: Caffe

Pre-trained on: PASCAL VOC dataset

Detectable Classes (20):

Person

Car

Bus

Bicycle

Dog

Cat

Bottle

Chair

And more...

MobileNet SSD is optimized for real-time applications and performs efficiently even on systems without a GPU.

