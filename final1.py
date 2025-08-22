import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import pandas as pd
import tempfile
import os

st.set_page_config(page_title="YOLOv8 Object Detection", page_icon="🤖", layout="wide")

st.title("Object Detection with YOLOv8")
st.markdown("Upload an image or video, and let the YOLOv8 model detect objects.")

uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "png", "jpeg", "mp4", "avi"])

MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5 GB

if uploaded_file:
    file_size = uploaded_file.size
    st.write(f"File size: {file_size / (1024 * 1024):.2f} MB")

    if file_size > MAX_FILE_SIZE:
        st.error("File size exceeds the 5 GB limit. Please upload a smaller file.")
    else:
        try:
            model = YOLO(r"C:\Users\elaha\Desktop\project\YOLOv8s Model\YOLOv8s Model\Weights\best.pt")
        except Exception as e:
            st.error(f"Error loading YOLO model: {e}")
            st.stop()

        file_type = uploaded_file.name.split('.')[-1]

        if file_type in ["jpg", "png", "jpeg"]:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image")

                results = model(image)
                st.image(results[0].plot(), caption="Detected Objects")

                detections = results[0].boxes
                if detections:
                    df = pd.DataFrame({
                        "Class": detections.cls.cpu().numpy(),
                        "Confidence": detections.conf.cpu().numpy(),
                        "X_min": detections.xyxy[:, 0].cpu().numpy(),
                        "Y_min": detections.xyxy[:, 1].cpu().numpy(),
                        "X_max": detections.xyxy[:, 2].cpu().numpy(),
                        "Y_max": detections.xyxy[:, 3].cpu().numpy(),
                    })
                    st.subheader("Detection Results:")
                    st.dataframe(df)
                else:
                    st.write("No objects detected.")
            except Exception as e:
                st.error(f"Error processing the image: {e}")

        elif file_type in ["mp4", "avi"]:
            try:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                video_path = tfile.name

                st.video(video_path)
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                stframe = st.empty()
                progress_bar = st.progress(0)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = model(frame)
                    annotated_frame = results[0].plot()
                    stframe.image(annotated_frame)
                    progress = int(cap.get(cv2.CAP_PROP_POS_FRAMES) / total_frames * 100)
                    progress_bar.progress(progress)

                cap.release()
                st.success("Video processing complete.")
                os.unlink(tfile.name)
            except Exception as e:
                st.error(f"Error processing the video: {e}")
else:
    st.warning("Please upload a file to proceed.")

# Footer
st.write("\n**Note:** This application uses the YOLOv8n model to detect human actions from images or videos.")
