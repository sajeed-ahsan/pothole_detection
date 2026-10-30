import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import threading
import os
from ultralytics import YOLO
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

# --- Model and Severity Logic ---
try:
    # Updated to point to the YOLOv12 training output directory we defined earlier
    MODEL_PATH = 'runs/train/pothole_detection_yolov12_run1/weights/best.pt'
    model = YOLO(MODEL_PATH)
except Exception as e:
    messagebox.showerror("Model Loading Error", f"Failed to load YOLOv12 model from {MODEL_PATH}\nError: {e}")
    exit()

def get_severity(box_area):
    """Classifies pothole severity based on its normalized area."""
    LOW_SEVERITY_THRESHOLD = 0.005
    MEDIUM_SEVERITY_THRESHOLD = 0.02
    if box_area < LOW_SEVERITY_THRESHOLD:
        return 'Low'
    elif box_area < MEDIUM_SEVERITY_THRESHOLD:
        return 'Medium'
    else:
        return 'High'

SEVERITY_ORDER = {'Low': 1, 'Medium': 2, 'High': 3}

# --- GUI Application Class ---
class PotholeDetectorApp:
    def __init__(self, root):
        self.root = root
        # Updated title to reflect the YOLOv12 upgrade
        self.root.title("Pothole Detector - YOLOv12 with Tracking [Poobathy D]")
        self.root.geometry("1400x800")

        self.video_capture = None
        self.video_thread = None
        self.stop_event = threading.Event()
        self.tracked_potholes = {}

        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        right_frame = tk.Frame(main_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.btn_file = tk.Button(left_frame, text="Choose Image or Video", command=self.select_file)
        self.btn_file.pack(pady=5)
        
        self.image_label = tk.Label(left_frame, background='black')
        self.image_label.pack(fill=tk.BOTH, expand=True)

        results_label = tk.Label(right_frame, text="Detection Results", font=("Helvetica", 14, "bold"))
        results_label.pack(anchor="w", pady=(5, 2))
        
        self.results_text = tk.Text(right_frame, wrap=tk.WORD, font=("Consolas", 10), height=12)
        self.results_text.pack(fill=tk.X, expand=False)

        graph_label = tk.Label(right_frame, text="Severity Distribution", font=("Helvetica", 14, "bold"))
        graph_label.pack(anchor="w", pady=(10, 2))

        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.plot_aggregated_counts()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def select_file(self):
        self.stop_video_playback()
        filepath = filedialog.askopenfilename(filetypes=[("Media Files", "*.jpg *.jpeg *.png *.mp4")])
        if not filepath:
            return

        self.reset_aggregates()
        self.update_text("Processing... Please wait.")
        self.image_label.config(image='')
        
        file_extension = os.path.splitext(filepath)[1].lower()
        if file_extension in ['.jpg', '.jpeg', '.png']:
            threading.Thread(target=self.process_image, args=(filepath,), daemon=True).start()
        elif file_extension == '.mp4':
            self.video_capture = cv2.VideoCapture(filepath)
            self.stop_event.clear()
            self.video_thread = threading.Thread(target=self.process_video, daemon=True)
            self.video_thread.start()

    def reset_aggregates(self):
        self.tracked_potholes.clear()
        self.root.after(0, self.plot_aggregated_counts)

    def process_image(self, filepath):
        frame = cv2.imread(filepath)
        annotated_frame = self.run_tracking_on_frame(frame)
        self.root.after(0, self.update_ui, annotated_frame)

    def process_video(self):
        while not self.stop_event.is_set():
            if not self.video_capture or not self.video_capture.isOpened(): break
            ret, frame = self.video_capture.read()
            if not ret: break
            
            annotated_frame = self.run_tracking_on_frame(frame)
            self.root.after(0, self.update_ui, annotated_frame)
            time.sleep(1/60)

    def run_tracking_on_frame(self, frame):
        h, w, _ = frame.shape
        # Ultralytics track functionality remains fully compatible with YOLOv12
        results = model.track(frame, persist=True, verbose=False)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu()
            confs = results[0].boxes.conf.cpu()

            for track_id, box, conf in zip(track_ids, boxes, confs):
                x1, y1, x2, y2 = [int(c) for c in box]
                box_area_norm = ((x2 - x1) * (y2 - y1)) / (w * h)
                current_severity = get_severity(box_area_norm)
                
                if track_id in self.tracked_potholes:
                    if SEVERITY_ORDER[current_severity] > SEVERITY_ORDER[self.tracked_potholes[track_id]['severity']]:
                        self.tracked_potholes[track_id]['severity'] = current_severity
                else:
                    self.tracked_potholes[track_id] = {'severity': current_severity}

                final_severity = self.tracked_potholes[track_id]['severity']
                color = {'Low': (0, 255, 0), 'Medium': (0, 165, 255), 'High': (0, 0, 255)}.get(final_severity)
                label = f"ID {track_id}: {final_severity}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def update_ui(self, frame):
        total_count = len(self.tracked_potholes)
        severity_counts = Counter(p['severity'] for p in self.tracked_potholes.values())
        summary = (f"Total Unique Potholes: {total_count}\n\n"
                   f"Final Severity Counts:\n"
                   f"  Low:    {severity_counts['Low']}\n"
                   f"  Medium: {severity_counts['Medium']}\n"
                   f"  High:   {severity_counts['High']}\n")
        self.update_text(summary)
        self.plot_aggregated_counts(severity_counts)
        self.display_frame(frame)

    def display_frame(self, frame):
        h, w, _ = frame.shape
        label_w = self.image_label.winfo_width() or 640
        label_h = self.image_label.winfo_height() or 480
        scale = min(label_w / w, label_h / h) if w > 0 and h > 0 else 1
        if scale < 1: frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=img)
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def update_text(self, text):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state=tk.DISABLED)
        
    def plot_aggregated_counts(self, severity_counts=None):
        if severity_counts is None: severity_counts = Counter()
        self.ax.clear()
        severities = ['Low', 'Medium', 'High']
        counts = [severity_counts.get(s, 0) for s in severities]
        colors = ['green', 'orange', 'red']
        bars = self.ax.bar(severities, counts, color=colors)
        self.ax.set_title('Pothole Severity Distribution')
        self.ax.set_ylabel('Count')
        for bar in bars:
            yval = bar.get_height()
            if yval > 0: self.ax.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center')
        self.canvas.draw()

    def stop_video_playback(self):
        """Safely signals the video processing thread to stop."""
        self.stop_event.set()
        # We no longer wait with .join() as the thread is a daemon

    def on_closing(self):
        """Handle the window close event."""
        print("Closing application... cleaning up threads.")
    
        # Stop video processing loop
        self.stop_event.set()
        time.sleep(0.1)

        # Release video capture safely
        if self.video_capture is not None:
            try:
                self.video_capture.release()
            except:
                pass
        self.video_capture = None

        # Close matplotlib figure to avoid hanging threads
        plt.close('all')

        # Explicitly destroy the window
        self.root.destroy()

        # Force garbage collection and exit
        import gc, sys
        gc.collect()
        print("Application terminated.")
        sys.exit(0)

# --- Run the Application ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PotholeDetectorApp(root)
    root.mainloop()