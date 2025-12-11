import cv2
import numpy as np
import tempfile
import streamlit as st

st.set_page_config(page_title="Road Lane Detection", layout="wide")
st.title("🚗 Road Lane Detection (OpenCV)")

st.markdown("Upload a road video and the app will detect lanes in real-time.")

# File uploader
uploaded_file = st.file_uploader("Upload a road video", type=["mp4", "avi", "mov"])

# Function to process video and detect lanes
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[(0, height), (width, height),
                             (width, int(height * 0.6)), (0, int(height * 0.6))]])
        cv2.fillPoly(mask, polygon, 255)
        masked = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked, 2, np.pi/180, 100,
                                np.array([]), minLineLength=40, maxLineGap=50)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        out.write(frame)

    cap.release()
    out.release()

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    output_path = "output.mp4"
    process_video(tfile.name, output_path)

    st.success("✅ Processing complete! Here’s the output:")
    st.video(output_path)
