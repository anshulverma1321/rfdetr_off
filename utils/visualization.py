import supervision as sv
import cv2
import numpy as np

class Visualizer:
    def __init__(self, class_names=None):
        """
        Initializes visualization utilities.
        Args:
            class_names (list): List of class names mapping ID to string.
        """
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
        self.class_names = class_names
        
        # Load COCO classes if none provided, since RF-DETR defaults to COCO
        if self.class_names is None:
            try:
                from rfdetr.assets.coco_classes import COCO_CLASSES
                self.class_names = COCO_CLASSES
            except ImportError:
                print("Warning: Could not load COCO_CLASSES from rfdetr. Labels will show class IDs.")
                self.class_names = []

    def annotate(self, image, detections, filter_classes=None):
        """
        Draws bounding boxes and labels on an image.
        Args:
            image (np.ndarray): Original image (BGR).
            detections (sv.Detections): Supervision detections.
            filter_classes (list): List of class names or IDs to keep.
        Returns:
            np.ndarray: Annotated image.
        """
        if detections is None or len(detections) == 0:
            return image
            
        # Filter detections if requested
        if filter_classes is not None:
            mask = np.array([
                (self.class_names[cls_id] in filter_classes) if cls_id < len(self.class_names) else (cls_id in filter_classes)
                for cls_id in detections.class_id
            ])
            detections = detections[mask]
            
        if len(detections) == 0:
            return image

        # Generate labels
        labels = []
        for confidence, class_id in zip(detections.confidence, detections.class_id):
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
            labels.append(f"{class_name} {confidence:.2f}")

        # Ensure image is contiguous array for OpenCV
        annotated_image = image.copy()
        
        # Annotate
        annotated_image = self.box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = self.label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        
        return annotated_image

    def save_predictions_to_json(self, detections, output_path):
        """
        Saves raw predictions to a JSON file.
        """
        import json
        
        if detections is None:
            return
            
        results = []
        for i in range(len(detections)):
            bbox = detections.xyxy[i].tolist()
            conf = float(detections.confidence[i])
            cls_id = int(detections.class_id[i])
            cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            
            results.append({
                "bbox": bbox,
                "confidence": conf,
                "class_id": cls_id,
                "class_name": cls_name
            })
            
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
