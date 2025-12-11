#!/usr/bin/env python3
import argparse
import sys
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

DEFAULT_INPUT = "video_1.mp4"
DEFAULT_OUTPUT = "lane_detected_output.mp4"

def grayscale(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def canny(img_gray, low_threshold=50, high_threshold=150):
    return cv2.Canny(img_gray, low_threshold, high_threshold)

def gaussian_blur(img_gray, kernel_size=5):
    return cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)

def region_of_interest(img):
    h, w = img.shape[:2]
    polys = np.array([[
        (int(0.1 * w), h),
        (int(0.45 * w), int(0.6 * h)),
        (int(0.55 * w), int(0.6 * h)),
        (int(0.9 * w), h)
    ]], dtype=np.int32)
    mask = np.zeros_like(img)
    if img.ndim == 2:
        cv2.fillPoly(mask, polys, 255)
    else:
        cv2.fillPoly(mask, polys, (255,) * img.shape[2])
    return cv2.bitwise_and(img, mask)

def display_lines(img_bgr, lines):
    line_image = np.zeros_like(img_bgr)
    if lines is None:
        return line_image
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return line_image

def make_coordinates(img, params):
    slope, intercept = params
    if abs(slope) < 1e-3:
        slope = 1e-3 if slope >= 0 else -1e-3
    y1 = img.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2], dtype=int)

def average_slope_intercept(img, lines):
    if lines is None:
        return None
    left, right = [], []
    for l in lines:
        for x1, y1, x2, y2 in l:
            if x1 == x2:
                continue
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            if slope < 0:
                left.append((slope, intercept))
            else:
                right.append((slope, intercept))
    output_lines = []
    if left:
        left_avg = np.mean(left, axis=0)
        output_lines.append([make_coordinates(img, left_avg)])
    if right:
        right_avg = np.mean(right, axis=0)
        output_lines.append([make_coordinates(img, right_avg)])
    return output_lines if output_lines else None

# ---------- frame processing function for moviepy ----------
def process_frame(rgb_frame):
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    gray = grayscale(frame)
    blur = gaussian_blur(gray, kernel_size=5)
    edges = canny(blur, low_threshold=50, high_threshold=150)
    roi = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi, rho=2, theta=np.pi/180, threshold=100,
                            minLineLength=40, maxLineGap=5)
    avg_lines = average_slope_intercept(frame, lines)
    line_img = display_lines(frame, avg_lines)
    combo = cv2.addWeighted(frame, 0.8, line_img, 1.0, 1.0)
    return cv2.cvtColor(combo, cv2.COLOR_BGR2RGB)

def run(input_path, output_path, threads=4, preset='medium'):
    if not os.path.isfile(input_path):
        print(f"ERROR: input file not found: {input_path}")
        sys.exit(2)
    print("Opening input video:", input_path)
    clip = VideoFileClip(input_path)  # streaming
    fps = clip.fps
    w, h = int(clip.w), int(clip.h)
    print(f"Input resolution: {w}x{h}, fps: {fps}")
    print("Applying lane-detection to each frame...")
    processed = clip.fl_image(process_frame)
    print("Writing output video:", output_path)
    processed.write_videofile(output_path, codec='libx264', audio=False,
                              threads=threads, preset=preset)
    print("Done. Output saved as:", output_path)

def parse_args():
    p = argparse.ArgumentParser(description="Lane detection (moviepy streaming frames)")
    p.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Input video file (.mp4)")
    p.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="Output mp4 filename")
    p.add_argument("--threads", type=int, default=4, help="FFmpeg/moviepy thread count")
    p.add_argument("--preset", default="medium", help="Encoding preset for libx264 (ultrafast|medium|slow...)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output, threads=args.threads, preset=args.preset)
