import streamlit as st
import cv2
import numpy as np

from PIL import Image
from ultralytics import YOLO

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🚧",
    layout="wide"
)

# -------------------------------
# LOAD MODEL
# -------------------------------

@st.cache_resource
def load_model():

    model = YOLO("best.pt")

    return model

model = load_model()

# -------------------------------
# APPLICATION HEADER
# -------------------------------

st.title("🚧 AI Road Damage Detection")

st.write(
    "Upload road images to detect potholes "
    "using YOLO11."
)

# -------------------------------
# SIDEBAR SETTINGS
# -------------------------------

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.4
)

iou = st.sidebar.slider(
    "NMS IoU Threshold",
    0.1,
    0.9,
    0.45
)

# -------------------------------
# IMAGE UPLOAD
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    # Display original image
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(image, use_container_width=True)

    # -------------------------------
    # MODEL PREDICTION
    # -------------------------------

    if st.button("Detect Potholes"):

        with st.spinner("Analyzing road image..."):

            results = model.predict(
                source=image,
                conf=confidence,
                iou=iou,
                imgsz=800,
                verbose=False
            )

            result = results[0]

            # Get detections
            boxes = result.boxes

            count = len(boxes)

            # Annotated image
            annotated = result.plot()

            annotated = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

        # -------------------------------
        # DISPLAY RESULTS
        # -------------------------------

        with col2:

            st.subheader("Detection Results")

            st.image(
                annotated,
                use_container_width=True
            )

        st.divider()

        # Metrics
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Potholes",
            count
        )

        col2.metric(
            "Confidence Threshold",
            f"{confidence:.2f}"
        )

        if count > 0:

            avg_conf = boxes.conf.mean().item()

        else:

            avg_conf = 0.0

        col3.metric(
            "Average Confidence",
            f"{avg_conf:.2%}"
        )

        # -------------------------------
        # DETECTION DETAILS
        # -------------------------------

        st.subheader("Detection Details")

        if count > 0:

            for i, box in enumerate(boxes):

                score = box.conf.item()

                st.write(
                    f"Pothole {i+1}: "
                    f"Confidence = {score:.2%}"
                )

        else:

            st.info("No potholes detected.")

        # -------------------------------
        # DOWNLOAD RESULT
        # -------------------------------

        _, buffer = cv2.imencode(
            ".jpg",
            cv2.cvtColor(
                annotated,
                cv2.COLOR_RGB2BGR
            )
        )

        st.download_button(
            label="Download Annotated Image",
            data=buffer.tobytes(),
            file_name="road_damage_detection.jpg",
            mime="image/jpeg"
        )
# =====================================
# VIDEO DETECTION
# =====================================

import tempfile
import os

st.divider()

st.header("🎥 Road Damage Detection - Video")

uploaded_video = st.file_uploader(
    "Upload Road Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_video:

    if st.button("Start Video Detection"):

        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_input:

            temp_input.write(uploaded_video.read())
            input_path = temp_input.name

        cap = cv2.VideoCapture(input_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if fps <= 0:
            fps = 25

        output_path = tempfile.mktemp(suffix=".mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        progress = st.progress(0)

        status = st.empty()

        frame_count = 0
        total_detections = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            # YOLO prediction
            result = model.predict(
                frame,
                conf=confidence,
                iou=iou,
                imgsz=800,
                verbose=False
            )[0]

            # Draw bounding boxes
            annotated_frame = result.plot()

            out.write(annotated_frame)

            total_detections += len(result.boxes)

            frame_count += 1

            progress.progress(
                min(frame_count / max(total_frames, 1), 1.0)
            )

            status.write(
                f"Processing frame {frame_count}/{total_frames}"
            )

        cap.release()
        out.release()

        os.remove(input_path)

        st.success("Video processing completed!")

        # Display processed video
        with open(output_path, "rb") as video_file:

            video_bytes = video_file.read()

        st.video(video_bytes)

        # Display metrics
        col1, col2 = st.columns(2)

        col1.metric(
            "Frames Processed",
            frame_count
        )

        col2.metric(
            "Total Detections Across Frames",
            total_detections
        )

        st.download_button(
            "Download Processed Video",
            data=video_bytes,
            file_name="road_damage_detected.mp4",
            mime="video/mp4"
        )

        os.remove(output_path)