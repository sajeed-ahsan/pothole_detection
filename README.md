# Pothole Detection and Analysis GUI

This is a Python-based desktop application that uses a YOLOv8 object detection model to identify, track, and analyze potholes in images and videos.

## Features

- **Image & Video Analysis:** Process single images or video files through a user-friendly interface.
- **Real-Time Detection:** Draws color-coded bounding boxes around potholes directly on the media.
- **Severity Classification:** Categorizes each pothole as 'Low', 'Medium', or 'High' based on its relative size.
- **Object Tracking:** Assigns a unique ID to each pothole in a video, preventing duplicate counts across frames and ensuring accurate statistics.
- **Aggregated Statistics:** The dashboard provides a running total of unique potholes and their final severity distribution for an entire video.
- **Visual Dashboard:** Displays the processed image/video alongside a live-updating text summary and a Matplotlib bar chart of the severity counts.

## Setup Instructions

1.  **Clone the Repository**
    ```
    git clone <your-github-repo-url>
    cd <your-repo-folder-name>
    ```

2.  **Create and Activate a Virtual Environment**
    ```
    # Create the environment
    python -m venv pothole_env

    # Activate on Windows
    .\pothole_env\Scripts\activate
    ```

3.  **Install Dependencies**
    With the virtual environment activated, install all required packages from the `requirements.txt` file.
    ```
    python -m pip install -r requirements.txt
    ```

## How to Run

With the virtual environment activated, run the application using the following command:

python app.py

This will launch the Tkinter GUI. Use the "Choose Image or Video" button to select a file and begin the analysis.

