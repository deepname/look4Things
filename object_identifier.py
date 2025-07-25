import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
import logging

# Import specialized models
from models.clip_model import CLIPModel
from models.blip_model import BLIPModel
from models.yolo_model import YOLOModel
from models.ollama_model import OllamaModel

# Import utilities
from utils.visualization import Visualizer
from utils.file_utils import FileManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectIdentifier:
    """
    Optimized and modular object identifier using multiple AI models:
    - CLIP for image-text similarity
    - BLIP for image captioning
    - YOLO for precise object detection
    - Ollama for advanced reasoning
    """
    
    def __init__(self, device: str = None):
        """Initialize all models and utilities"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize specialized models
        self.clip_model = CLIPModel(device=self.device)
        self.blip_model = BLIPModel(device=self.device)
        self.yolo_model = YOLOModel()
        self.ollama_model = OllamaModel()
        
        # Initialize utilities
        self.visualizer = Visualizer()
        self.file_manager = FileManager()
        
        logger.info("ObjectIdentifier initialized successfully!")
    
    def identify_comprehensive(self, image_path: str, candidate_objects: List[str] = None) -> Dict:
        """
        Comprehensive object identification using all available models
        
        Args:
            image_path: Path to image file
            candidate_objects: Optional list of objects to look for
            
        Returns:
            Complete analysis results
        """
        try:
            # Load image
            image_pil = Image.open(image_path).convert('RGB')
            
            results = {
                'image_path': image_path,
                'timestamp': self.device,
            }
            
            # Generate caption with BLIP
            logger.info("Generating image caption...")
            caption = self.blip_model.generate_caption(image_pil)
            results['caption'] = caption
            
            # Identify objects with CLIP
            if candidate_objects and self.clip_model.is_available():
                logger.info("Identifying objects with CLIP...")
                clip_results = self.clip_model.identify_objects(image_pil, candidate_objects)
                results['clip_objects'] = clip_results
            
            # Detect precise objects with YOLO
            yolo_detections = []
            if candidate_objects and self.yolo_model.is_available():
                logger.info("Detecting objects with YOLO...")
                yolo_detections = self.yolo_model.detect_objects(image_path, candidate_objects)
                results['yolo_detections'] = yolo_detections
            
            # Advanced analysis with Ollama
            logger.info("Performing advanced analysis...")
            if 'clip_objects' in results:
                analysis = self.ollama_model.analyze(caption, results['clip_objects'])
                results['advanced_analysis'] = analysis
            else:
                results['advanced_analysis'] = "Advanced analysis not available"
            
            # Create visualization with bounding boxes
            if yolo_detections or results.get('clip_objects'):
                output_image_path, box_info = self.visualizer.create_bounding_box_image(
                    image_path,
                    yolo_detections=yolo_detections,
                    clip_objects=results.get('clip_objects'),
                    output_dir="result"
                )
                results['output_image_path'] = output_image_path
                results['bounding_boxes'] = box_info
            
            # Additional metadata
            results['models_used'] = {
                'clip': self.clip_model.is_available(),
                'blip': self.blip_model.is_available(),
                'yolo': self.yolo_model.is_available(),
                'ollama': True  # Always available
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive identification failed: {e}")
            return {'error': str(e)}
    
    def save_results_to_folder(self, results: Dict, output_dir: str = "result") -> Dict[str, str]:
        """
        Save comprehensive results to folder
        
        Args:
            results: Results from identify_comprehensive
            output_dir: Directory to save all results
            
        Returns:
            Dictionary with paths to saved files
        """
        return self.file_manager.save_results_to_folder(results, output_dir)
    
    def visualize_results(self, results: Dict, save_path: str = None):
        """
        Visualize identification results
        
        Args:
            results: Results from identify_comprehensive
            save_path: Optional path to save visualization
        """
        self.visualizer.create_results_plot(results, save_path)
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get status of all models"""
        return {
            'CLIP': self.clip_model.is_available(),
            'BLIP': self.blip_model.is_available(),
            'YOLO': self.yolo_model.is_available(),
            'Ollama': True,
            'Device': self.device
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize identifier
    identifier = ObjectIdentifier()
    
    # Print model status
    print("Object Identifier initialized successfully!")
    print("Available models:")
    status = identifier.get_model_status()
    for model, available in status.items():
        if model != 'Device':
            print(f"- {model}: {'✓' if available else '✗'}")
    print(f"- Device: {status['Device']}")
    
    # Example usage (uncomment to test)
    # results = identifier.identify_comprehensive("path/to/your/image.jpg", ["scissors", "pen"])
    # identifier.save_results_to_folder(results)
    # identifier.visualize_results(results)
