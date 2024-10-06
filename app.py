import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import tempfile

# Load the ONNX model
@st.cache_resource
def load_model():
    return ort.InferenceSession("model.onnx")

ort_session = load_model()

def preprocess_image(image, target_size=(640, 640)):
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Resize image
    image = cv2.resize(image, target_size)
    # Normalize
    image = image.astype(np.float32) / 255.0
    # Transpose for ONNX input
    image = np.transpose(image, (2, 0, 1))
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def postprocess_results(output, image_shape, confidence_threshold=0.25, iou_threshold=0.45):
    # Handle different possible output formats
    if isinstance(output, (list, tuple)):
        predictions = output[0]
    elif isinstance(output, np.ndarray):
        predictions = output
    else:
        raise ValueError(f"Unexpected output type: {type(output)}")

    # Reshape if necessary
    if len(predictions.shape) == 4:
        predictions = predictions.squeeze((0, 1))
    elif len(predictions.shape) == 3:
        predictions = predictions.squeeze(0)

    # Extract boxes, scores, and class_ids
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    class_ids = predictions[:, 5]

    # Filter by confidence
    mask = scores > confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    # Convert boxes from [x, y, w, h] to [x1, y1, x2, y2]
    boxes[:, 2:] += boxes[:, :2]

    # Scale boxes to image size
    boxes[:, [0, 2]] *= image_shape[1]
    boxes[:, [1, 3]] *= image_shape[0]

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), confidence_threshold, iou_threshold)

    results = []
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        x1, y1, x2, y2 = map(int, box)
        results.append((x1, y1, x2, y2, float(score), int(class_id)))

    return results

def process_image(image):
    orig_image = image.copy()
    processed_image = preprocess_image(image)
    
    # Run inference
    inputs = {ort_session.get_inputs()[0].name: processed_image}
    outputs = ort_session.run(None, inputs)
    
    results = postprocess_results(outputs, image.shape)
    
    # Draw bounding boxes on the image
    for x1, y1, x2, y2, score, class_id in results:
        cv2.rectangle(orig_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"License Plate: {score:.2f}"
        cv2.putText(orig_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create a temporary file to store the processed video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_image(frame)
        out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
    
    cap.release()
    out.release()
    
    return temp_file.name

st.title("License Plate Detection")

uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == "image":
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect License Plates"):
            processed_image = process_image(np.array(image))
            st.image(processed_image, caption="Processed Image", use_column_width=True)
    
    elif file_type == "video":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        st.video(tfile.name)
        
        if st.button("Detect License Plates"):
            processed_video = process_video(tfile.name)
            st.video(processed_video)

st.write("Upload an image or video to detect license plates.")