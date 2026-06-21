import cv2
import numpy as np
import time

# ─── Configuration ────────────────────────────────────────────────
ALPHA      = 0.6
BETA       = 0.4
T_MAX      = 0.1
VIDEO_PATH = "video.mp4"  
LINE_Y     = 250          

# ─── Frame Analysis ───────────────────────────────────────────────
def analyse_frame(frame, prev_frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    noise_level = np.var(gray)
    motion_intensity = 0
    if prev_frame is not None:
        diff = cv2.absdiff(gray, cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
        motion_intensity = np.mean(diff)
    return noise_level, motion_intensity

# ─── Heuristic Preprocessing ──────────────────────────────────────
def heuristic_preprocess(frame, noise_level):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if noise_level > 1500:
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
    else:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

def heuristic_threshold(motion_intensity):
    return 25 if motion_intensity > 20 else 40

# ─── Object Detection ─────────────────────────────────────────────
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=200, varThreshold=50, detectShadows=True
)

def detect_objects(frame):
    fg_mask = bg_subtractor.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 800:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        detections.append((cx, cy, x, y, w, h))
    return detections, fg_mask

# ─── Centroid Tracker ─────────────────────────────────────────────
class CentroidTracker:
    def __init__(self, max_distance=80):
        self.next_id  = 0
        self.objects  = {}
        self.max_dist = max_distance
        self.errors   = 0

    def update(self, detections):
        new_objects   = {}
        det_centroids = [(d[0], d[1]) for d in detections]
        if not self.objects:
            for cx, cy in det_centroids:
                new_objects[self.next_id] = (cx, cy)
                self.next_id += 1
        else:
            old_ids       = list(self.objects.keys())
            old_centroids = list(self.objects.values())
            matched_old   = set()
            for ni, (nx, ny) in enumerate(det_centroids):
                best_dist, best_oid = float('inf'), None
                for oi, (ox, oy) in enumerate(old_centroids):
                    d = np.hypot(nx - ox, ny - oy)
                    if d < best_dist:
                        best_dist, best_oid = d, oi
                if best_dist < self.max_dist and best_oid not in matched_old:
                    new_objects[old_ids[best_oid]] = (nx, ny)
                    matched_old.add(best_oid)
                else:
                    if best_dist >= self.max_dist:
                        self.errors += 1
                    new_objects[self.next_id] = (nx, ny)
                    self.next_id += 1
        self.objects = new_objects
        return new_objects

# ─── Line Counter ─────────────────────────────────────────────────
class LineCounter:
    def __init__(self, line_y):
        self.line_y  = line_y
        self.prev_y  = {}
        self.counted = set()
        self.count   = 0

    def update(self, tracked_objects):
        for obj_id, (cx, cy) in tracked_objects.items():
            if obj_id in self.prev_y:
                py = self.prev_y[obj_id]
                if (py < self.line_y <= cy or cy < self.line_y <= py):
                    if obj_id not in self.counted:
                        self.count += 1
                        self.counted.add(obj_id)
            self.prev_y[obj_id] = cy

# ─── Cost Function ────────────────────────────────────────────────
def compute_cost(errors, total_objects, elapsed):
    e_ratio = min(errors / total_objects, 1.0) if total_objects > 0 else 0
    t_ratio = min(elapsed / T_MAX, 1.0)
    return ALPHA * e_ratio + BETA * t_ratio

# ─── Main ─────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {VIDEO_PATH}"); return

    # Get video properties for display
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} | {fps:.1f} FPS | {total_frames} frames")

    tracker      = CentroidTracker()
    line_counter = LineCounter(LINE_Y)
    prev_frame   = None
    total_N      = 0
    frame_num    = 0

    print("Video started. Press Q to quit, SPACE to pause.")
    paused = False

    while True:
        if not paused:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Video ended."); break

            frame_num += 1
            frame = cv2.resize(frame, (640, 480))

            noise_level, motion_intensity = analyse_frame(frame, prev_frame)
            heuristic_preprocess(frame, noise_level)
            heuristic_threshold(motion_intensity)

            detections, fg_mask = detect_objects(frame)
            total_N += len(detections)

            tracked = tracker.update(detections)
            line_counter.update(tracked)

            elapsed = time.time() - t_start
            cost    = compute_cost(tracker.errors, max(total_N, 1), elapsed)

            # Progress bar
            progress = int((frame_num / total_frames) * 200) if total_frames > 0 else 0
            cv2.rectangle(frame, (10, 455), (210, 470), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 455), (10 + progress, 470), (0, 255, 100), -1)
            cv2.putText(frame, f"Frame {frame_num}/{total_frames}", (220, 468),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Draw
            cv2.line(frame, (0, LINE_Y), (640, LINE_Y), (0, 255, 255), 2)
            for obj_id, (cx, cy) in tracked.items():
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID {obj_id}", (cx - 10, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            for d in detections:
                cv2.rectangle(frame, (d[2], d[3]), (d[2]+d[4], d[3]+d[5]), (255, 100, 0), 2)

            cv2.putText(frame, f"Count : {line_counter.count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Cost  : {cost:.3f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(frame, f"Noise : {noise_level:.0f}  Motion: {motion_intensity:.1f}",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, "VIDEO MODE", (490, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            cv2.imshow("Video - Object Tracking & Counting", frame)
            cv2.imshow("Foreground Mask", fg_mask)
            prev_frame = frame.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):        # spacebar to pause/resume
            paused = not paused
            print("Paused" if paused else "Resumed")

    print(f"\n── Video Session Results ──")
    print(f"Video File   : {VIDEO_PATH}")
    print(f"Frames Proc. : {frame_num}")
    print(f"Final Count  : {line_counter.count}")
    print(f"Total Errors : {tracker.errors}")
    print(f"Final Cost   : {compute_cost(tracker.errors, max(total_N,1), T_MAX):.4f}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()