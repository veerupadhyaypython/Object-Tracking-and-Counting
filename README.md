# Object-Tracking-and-Counting
Real-time object tracking and counting system using a Greedy Heuristic with Rule-Based Adaptive Decision Making, built with Python and OpenCV — dynamically adjusts preprocessing, thresholding, and tracking based on live frame analysis to deliver accurate, low-latency object counts without ML overhead.
# Real-Time Object Tracking and Counting

A real-time object tracking and counting system built using a **Greedy Heuristic with Rule-Based Adaptive Decision Making**. Unlike traditional fixed-pipeline computer vision approaches, this system dynamically adapts its preprocessing, thresholding, and tracking parameters based on live frame analysis — without relying on machine learning models.

> Developed as a Design and Analysis of Algorithms (DAA) Lab Mini Project.

## Overview

Object tracking is widely used in surveillance, traffic analysis, and automation, but traditional methods (background subtraction, optical flow, ML-based tracking) follow fixed pipelines that struggle with noise, lighting changes, and occlusion. This project addresses that limitation with a **greedy, frame-independent heuristic approach**: at every frame, the system makes the best immediate decision based only on current information — no lookahead, no backtracking.

## Key Features

**Adaptive Preprocessing** — adjusts Gaussian blur kernel size (3x3 / 7x7) based on pixel-variance noise levels
**Dynamic Thresholding** — sets motion threshold (40 / 25) based on frame-difference motion intensity
**Minimum Area Filtering** — discards contours below 800px to suppress noise
**Centroid-Based Tracking** — nearest-neighbour matching (distance < 80px) to maintain object identity across frames
**Virtual Trip-Wire Counting** — line-crossing logic to prevent duplicate counts
**Cost Function Evaluation** — normalized metric balancing tracking accuracy and processing speed

## Heuristic Rules

| Rule | Input Signal | Decision |
|---|---|---|
| Adaptive Preprocessing | Pixel variance | Kernel size: 3x3 (low noise) / 7x7 (high noise) |
| Dynamic Thresholding | Frame difference | Threshold: 40 (slow) / 25 (fast motion) |
| Minimum Area Filtering | Contour area | Keep if area > 800px |
| Nearest Neighbour Matching | Centroid distance | Same object if distance < 80px |
| Virtual Trip-Wire Counting | Y-coordinate crossing | Count once per object |

## Cost Function

System performance is evaluated using:
C = α(E/N) + β(T/Tmax)

| Symbol | Meaning |
|---|---|
| E | Number of tracking errors |
| N | Total detected objects |
| T | Processing time per frame |
| Tmax | Max allowable processing time (0.1s) |
| α | Accuracy weight (0.6) |
| β | Speed weight (0.4) |

- **Cmin = 0** → no tracking errors, minimal processing time
- **Cmax = 1** → all objects mismatched, T = Tmax

## Tech Stack

- **Python 3.11**
- **OpenCV (cv2) 4.13** — video capture, image processing, contour detection
- **NumPy 2.4** — pixel variance computation, array operations
- **MOG2 Background Subtractor** — foreground object detection

## System Workflow

1. Capture video frames (webcam or pre-recorded video)
2. Analyze frame noise (`np.var`) and motion intensity (`cv2.absdiff`)
3. Apply heuristic rules to set preprocessing/threshold values
4. Detect objects via MOG2 + contour detection
5. Track objects using centroid-based nearest-neighbour matching
6. Count objects via virtual trip-wire line-crossing logic
7. Display bounding boxes, object IDs, count, and cost function value

## Key Components

| Function / Class | Purpose |
|---|---|
| `analyse_frame()` | Computes noise level and motion intensity |
| `heuristic_preprocess()` | Adaptive Gaussian blur based on noise |
| `heuristic_threshold()` | Dynamic threshold based on motion |
| `detect_objects()` | MOG2 + contour detection + bounding boxes |
| `CentroidTracker` | Nearest-neighbour matching across frames |
| `LineCounter` | Virtual trip-wire line-crossing counter |
| `compute_cost()` | Evaluates normalized cost function |

## Results

Tested on a pre-recorded pedestrian video (1280x720, 60fps):

- ✅ Accurate moving-object detection via MOG2 background subtraction
- ✅ Minimal ID switches in centroid-based tracking
- ✅ No duplicate counting (trip-wire logic)
- ✅ Real-time processing within Tmax = 0.1s/frame at 640x480
- ✅ Low overall cost value — strong accuracy/speed balance

A secondary **Foreground Mask** window visualizes detected motion blobs in real time, providing visual confirmation of the adaptive heuristics at work.

## Authors

- Veer Upadhyay – 245819272
- Yanis Singla – 245819368

Class ECM-B, DAA Lab Mini Project

## References

- [OpenCV Documentation](https://docs.opencv.org)
- Gonzalez, R. C., and Woods, R. E., *Digital Image Processing*, Pearson Education
- Szeliski, R., *Computer Vision: Algorithms and Applications*, Springer
- [NumPy Documentation](https://numpy.org/doc)
