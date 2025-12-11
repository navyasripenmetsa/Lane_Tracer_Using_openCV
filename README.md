# Lane_Tracer_Using_openCV


This project implements a **traditional computer-vision–based lane detection system** using OpenCV. It processes dashcam footage to identify road lane boundaries, simulating the core of a **basic lane-following module** used in autonomous driving systems.

---

##  **Input & Output**

### **Input Video**
- **File:** `solidWhiteRight.mp4`,  `test_video_input.mp4`
- **Source:** Udacity Self-Driving Car Dataset  
- Clean daytime driving clip containing clear white lane markings.

### **Output Video**
Processed output with lane overlays:  
**`lane_detected_output.mp4`** ,  **`test_video_output.mp4`**

---


##  **Pipeline Overview**

Each video frame goes through a series of classical image-processing steps:

### • **Grayscale Conversion**
Simplifies pixel information for easier edge detection.

### • **Gaussian Blur**
Smoothens noise and prevents false edges.

### • **Canny Edge Detection**
Extracts significant edges from the scene.

### • **Region of Interest (ROI) Masking**
Focuses only on the relevant road area.

### • **Hough Line Transform**
Detects straight lines that correspond to lane boundaries.

### • **Lane Averaging & Extrapolation**
Identifies left/right lane segments and extends them across the frame.

### • **Overlay Rendering**
Draws the detected lane lines on top of the original video.

---

##  **Technologies Used**
- **Python 3**
- **OpenCV**
- **NumPy**

---

##  **Requirements**
opencv-python
numpy




## Project Structure

| File                          | Description                                           |
|-------------------------------|-------------------------------------------------------|
| `lane_detection.py`           | Main script that processes the video                 |
| `solidWhiteRight.mp4`         | Input driving video                                  |
| `lane_detected_output.mp4`    | Output video with detected lanes                     |
| `requirements.txt`            | Python libraries used                                |
| `README.md`                   | Project documentation                                |
| `test_video_output.mp4`       | input driving video                                  |
| `test_video_output.mp4`       | output driving video                                 |
 ---

 Future Enhancements

Support curved lanes using polynomial fitting

Integrate real-time webcam lane detection

Use color masks to detect yellow lanes better

Deploy in embedded systems or mobile devices
