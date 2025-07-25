from ultralytics import YOLO
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class YOLOModel:
    """YOLO model for object detection"""
    
    def __init__(self):
        self.yolo_model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            # Load YOLOv8 model (will download automatically if not present)
            self.yolo_model = YOLO('yolov8n.pt')  # Using nano version for speed
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.warning(f"YOLO model not available: {e}")
            self.yolo_model = None
    
    def is_available(self) -> bool:
        """Check if YOLO model is available"""
        return self.yolo_model is not None
    
    def detect_objects(self, image_path: str, target_classes: List[str] = None) -> List[Dict]:
        """
        Detect multiple objects using YOLO and return individual bounding boxes
        
        Args:
            image_path: Path to the image
            target_classes: List of target class names to filter (e.g., ['scissors'])
            
        Returns:
            List of detected objects with bounding boxes and confidence scores
        """
        if not self.is_available():
            return []
        
        try:
            # Run YOLO detection
            results = self.yolo_model(image_path)
            
            detected_objects = []
            
            # YOLO class names mapping for common objects
            yolo_class_mapping = {
                'scissors': 76,  # COCO class ID for scissors
                'knife': 49,     # Sometimes scissors might be detected as knife
                'spoon': 50,     # Kitchen utensils
                'fork': 48       # Kitchen utensils
            }
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class ID and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = result.names[class_id]
                        
                        # Filter by target classes if specified
                        if target_classes:
                            # Check if the detected class matches any target class
                            matches_target = False
                            for target in target_classes:
                                if (target.lower() in class_name.lower() or 
                                    class_name.lower() in target.lower() or
                                    (target.lower() == 'tijeras' and class_name.lower() == 'scissors') or
                                    (target.lower() == 'scissors' and class_id == yolo_class_mapping.get('scissors'))):
                                    matches_target = True
                                    break
                            
                            if not matches_target:
                                continue
                        
                        # Get bounding box coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        detected_objects.append({
                            'class_name': class_name,
                            'confidence': confidence,
                            'bounding_box': {
                                'left': int(x1),
                                'top': int(y1),
                                'right': int(x2),
                                'bottom': int(y2)
                            },
                            'class_id': class_id
                        })
            
            # Sort by confidence (highest first)
            detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f"YOLO detected {len(detected_objects)} objects")
            return detected_objects
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []
