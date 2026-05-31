import os
import torch
import json
import supervision as sv
from PIL import Image

class RFDETRInference:
    def __init__(self, size="M", config_path="configs/config.json"):
        """
        Initializes the RF-DETR Model.
        Args:
            size (str): N, S, M, L, XL, 2XL
            config_path (str): path to config file
        """
        self.size = size.upper()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load config if exists
        self.config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
                
        # Set cache dir for offline execution if specified
        cache_dir = self.config.get("model", {}).get("cache_dir", "./models/pretrained")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir # Force huggingface hub to use local cache
        
        # Initialize the appropriate model class
        self.model = self._load_model()
        
    def _load_model(self):
        print(f"Loading RF-DETR Size: {self.size} on {self.device}...")
        try:
            if self.size == "N":
                from rfdetr import RFDETRNano
                return RFDETRNano()
            elif self.size == "S":
                from rfdetr import RFDETRSmall
                return RFDETRSmall()
            elif self.size == "M":
                from rfdetr import RFDETRMedium
                return RFDETRMedium()
            elif self.size == "L":
                from rfdetr import RFDETRLarge
                return RFDETRLarge()
            elif self.size == "XL":
                from rfdetr import RFDETRXLarge
                return RFDETRXLarge()
            elif self.size == "2XL":
                from rfdetr import RFDETR2XLarge
                return RFDETR2XLarge()
            else:
                raise ValueError(f"Unsupported model size: {self.size}. Choose from N, S, M, L, XL, 2XL")
        except ImportError as e:
            raise ImportError("Could not import rfdetr package. Ensure it is installed via 'pip install rfdetr'") from e

    def predict(self, image, threshold=None):
        """
        Runs inference on an image.
        Args:
            image (str, np.ndarray, PIL.Image): Input image.
            threshold (float): Confidence threshold.
        Returns:
            sv.Detections: Supervision detections object.
        """
        if threshold is None:
            threshold = self.config.get("inference", {}).get("default_confidence_threshold", 0.5)
            
        # The rfdetr package's predict method handles various inputs
        # It returns a supervision Detections object directly
        try:
            detections = self.model.predict(image, threshold=threshold)
            return detections
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
