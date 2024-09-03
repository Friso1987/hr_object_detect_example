# hr_object_detect_example

## Description

`hr_object_detect_example` is an object detection tool created as an example for educational purposes. It utilizes TensorFlow and OpenCV to perform real-time object detection using a webcam. The tool can detect multiple objects in the video stream and logs the detected objects along with the time of detection to a CSV file.

## Installation

Follow these steps to set up the project on your local machine:

### Prerequisites

- **Python 3.x**: Make sure Python is installed on your system. You can download it from [python.org](https://www.python.org/).
- **pip**: Python's package installer. It is included with Python 3.x.

### Dependencies

You can install the required dependencies using `pip`:

```bash
pip install opencv-python-headless tensorflow
```
## Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/hr_object_detect_example.git
cd hr_object_detect_example
```
## Usage

### Prepare the Environment:
Ensure that your webcam is connected and functioning. Place the `labelmap.pbtxt` file in the same directory as your `main.py` script.

### Run the Script:
To start the object detection, run the following command:

```bash
python main.py
```

The tool will open a window displaying the video stream from your webcam with detected objects highlighted. It will also log the detected objects along with the time of detection to object_detections.csv.
