# License Plate Detection

This project is a Streamlit web application that uses a custom YOLO model to detect license plates in images and videos. The application is designed to be deployed on Hugging Face Spaces but can also be run locally.

## Features

- Upload and process both images and videos
- Real-time license plate detection
- User-friendly interface built with Streamlit
- Uses a custom YOLO model converted to ONNX format for inference

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/license-plate-detection.git
   cd license-plate-detection
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Ensure your ONNX model file is named `model.onnx` and is in the project root directory.

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your web browser and go to `http://localhost:8501` to use the application.

4. Upload an image or video file and click the "Detect License Plates" button to process it.

## Project Structure

- `app.py`: Main application file containing the Streamlit interface and processing logic
- `model.onnx`: YOLO model in ONNX format for license plate detection
- `requirements.txt`: List of Python dependencies
- `README.md`: This file, containing project information and instructions

## Deploying to Hugging Face Spaces

To deploy this project on Hugging Face Spaces:

1. Create a new Space on Hugging Face.
2. Set the Space settings:
   - Python version: 3.9 or higher
   - SDK: Streamlit
   - App file: app.py
3. Upload your `app.py`, `model.onnx`, and `requirements.txt` files to the Space.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

