from ultralytics import YOLO
import os
import cv2 # Import OpenCV

def get_severity(box_area):
    """
    Classifies the severity of a pothole based on its bounding box area.

    Args:
        box_area (float): The normalized area of the bounding box (0 to 1).

    Returns:
        str: The severity category ('Low', 'Medium', or 'High').
    """
    # These thresholds are adjustable. They represent the percentage of the image area.
    LOW_SEVERITY_THRESHOLD = 0.005  # Area < 0.5% of the image
    MEDIUM_SEVERITY_THRESHOLD = 0.02 # Area < 2% of the image

    if box_area < LOW_SEVERITY_THRESHOLD:
        return 'Low'
    elif box_area < MEDIUM_SEVERITY_THRESHOLD:
        return 'Medium'
    else:
        return 'High'

def main():
    # --- Configuration ---
    # Path to your trained model
    model_path = 'runs/train/pothole_detection_run12/weights/best.pt'
    
    # Test image or video path - change this to your test file
    # Example for an image: 'dataset/valid/images/pothole-image-1.jpg'
    # Example for a video: 'path/to/your/test_video.mp4'
    test_source = 'D:/Projects/Pothole/dataset/test/images/pothole-in-2.png'  # <<< CHANGE THIS PATH
    
    # Output directory for results
    output_dir = 'prediction_results_with_severity'
    # ---------------------

    print("--- YOLO Pothole Detection with Severity ---")
    print(f"Loading model: {model_path}")
    
    # Load your trained model
    model = YOLO(model_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running inference on: {test_source}")
    print("Processing...")

    # Run inference on the source
    results = model(test_source, stream=True) # Use stream=True for memory efficiency

    # Process results stream
    for result in results:
        # Make a copy of the original image to draw on
        annotated_frame = result.orig_img.copy()
        
        if result.boxes is not None:
            # Get boxes, confidences, and class IDs
            for box in result.boxes:
                # Get coordinates
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                
                # Get confidence
                confidence = box.conf[0]
                
                # Get normalized area for severity calculation
                h, w = result.orig_shape
                box_width_norm = (x2 - x1) / w
                box_height_norm = (y2 - y1) / h
                box_area_norm = box_width_norm * box_height_norm

                # Get severity
                severity = get_severity(box_area_norm)
                
                # --- Print to Console ---
                print(f"Pothole Detected | Confidence: {confidence:.2f} | Severity: {severity}")

                # --- Draw on Image ---
                # Define color based on severity
                if severity == 'High':
                    color = (0, 0, 255) # Red
                elif severity == 'Medium':
                    color = (0, 165, 255) # Orange
                else:
                    color = (0, 255, 0) # Green
                
                # Create the label text
                label = f"Pothole: {confidence:.2f} (Severity: {severity})"
                
                # Draw the bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw the label background
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - label_height - 5), (x1 + label_width, y1), color, -1)
                
                # Put the label text
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # Save the annotated frame
        output_path = os.path.join(output_dir, os.path.basename(result.path))
        cv2.imwrite(output_path, annotated_frame)
    
    print("\n--- Inference Complete ---")
    print(f"Annotated results with severity saved in: '{output_dir}'")

if __name__ == '__main__':
    main()
