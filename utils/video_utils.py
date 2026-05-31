import cv2
import supervision as sv

class VideoProcessor:
    def __init__(self, source_path_or_id):
        """
        Initializes video processing.
        Args:
            source_path_or_id: String path to video file or integer for webcam.
        """
        self.source = source_path_or_id
        
    def process_video(self, inference_callback, visualization_callback, output_path=None):
        """
        Processes a video or webcam stream frame by frame.
        Args:
            inference_callback: Function that takes an image and returns detections.
            visualization_callback: Function that takes image & detections and returns annotated image.
            output_path: Path to save the output video. If None, won't save.
        """
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.source}")
            return
            
        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps == 0:
            fps = 30.0 # fallback for some webcams
            
        writer = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        print(f"Starting video processing from source: {self.source}")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Inference
                detections = inference_callback(frame)
                
                # Visualization
                annotated_frame = visualization_callback(frame, detections)
                
                if writer is not None:
                    writer.write(annotated_frame)
                    
                # Note: We don't use cv2.imshow() here to keep it strictly offline/headless compatible.
                # If running locally with GUI, user could uncomment:
                # cv2.imshow("RF-DETR", annotated_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                
        except KeyboardInterrupt:
            print("Video processing interrupted by user.")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            print("Video processing completed.")
