import os
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    """Handles visualization of detection results and bounding boxes"""
    
    def __init__(self):
        self.colors = [
            '#FF0000',  # Red
            '#00FF00',  # Green
            '#0000FF',  # Blue
            '#FFFF00',  # Yellow
            '#FF00FF',  # Magenta
            '#00FFFF',  # Cyan
            '#FFA500',  # Orange
            '#800080',  # Purple
            '#FFC0CB',  # Pink
            '#A52A2A'   # Brown
        ]
    
    def create_bounding_box_image(self, image_path: str, yolo_detections: List[Dict] = None, 
                                 clip_objects: Dict[str, float] = None, output_dir: str = "result", 
                                 threshold: float = 0.1) -> tuple:
        """
        Create an image with colored bounding boxes around detected objects
        
        Args:
            image_path: Path to the original image
            yolo_detections: List of YOLO detection results
            clip_objects: CLIP detection results (fallback)
            output_dir: Directory to save results
            threshold: Minimum confidence threshold
            
        Returns:
            Tuple of (output_image_path, box_info)
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Load the original image
            image = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            box_info = []
            
            # Use YOLO detections if available (precise bounding boxes)
            if yolo_detections:
                logger.info(f"Using YOLO detections for {len(yolo_detections)} objects")
                box_info = self._draw_yolo_boxes(draw, yolo_detections, threshold)
            
            # Fallback to CLIP-based estimated boxes
            elif clip_objects:
                logger.info("Using CLIP-based detection with estimated positions")
                box_info = self._draw_clip_boxes(draw, clip_objects, image.size, threshold)
            
            # Save the image with bounding boxes
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_image_path = os.path.join(output_dir, f"{base_name}_with_boxes.png")
            image.save(output_image_path)
            
            logger.info(f"Image with bounding boxes saved to: {output_image_path}")
            return output_image_path, box_info
            
        except Exception as e:
            logger.error(f"Failed to create bounding box image: {e}")
            return None, []
    
    def _draw_yolo_boxes(self, draw: ImageDraw.Draw, detections: List[Dict], threshold: float) -> List[Dict]:
        """Draw precise YOLO bounding boxes"""
        box_info = []
        
        for i, detection in enumerate(detections):
            if detection['confidence'] < threshold:
                continue
                
            color = self.colors[i % len(self.colors)]
            
            # Get precise bounding box from YOLO
            bbox = detection['bounding_box']
            left, top, right, bottom = bbox['left'], bbox['top'], bbox['right'], bbox['bottom']
            
            # Draw the bounding box
            draw.rectangle([left, top, right, bottom], outline=color, width=4)
            
            # Add label
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            self._draw_label(draw, label, left, top, color)
            
            # Store box info for JSON
            box_info.append({
                'object': detection['class_name'],
                'confidence': detection['confidence'],
                'bounding_box': bbox,
                'color': color,
                'detection_method': 'YOLO'
            })
        
        return box_info
    
    def _draw_clip_boxes(self, draw: ImageDraw.Draw, clip_objects: Dict[str, float], 
                        image_size: tuple, threshold: float) -> List[Dict]:
        """Draw estimated bounding boxes for CLIP detections"""
        box_info = []
        img_width, img_height = image_size
        
        # Filter and sort objects
        filtered_objects = {obj: score for obj, score in clip_objects.items() if score >= threshold}
        sorted_objects = sorted(filtered_objects.items(), key=lambda x: x[1], reverse=True)
        
        for i, (obj_name, confidence) in enumerate(sorted_objects[:len(self.colors)]):
            color = self.colors[i % len(self.colors)]
            
            # Create estimated bounding box
            box_size_factor = 0.3 + (confidence * 0.4)
            
            # Position boxes in different areas
            if i == 0:  # Most confident - center
                center_x, center_y = img_width // 2, img_height // 2
            elif i == 1:  # Second - upper left
                center_x, center_y = img_width // 4, img_height // 4
            elif i == 2:  # Third - upper right
                center_x, center_y = 3 * img_width // 4, img_height // 4
            elif i == 3:  # Fourth - lower left
                center_x, center_y = img_width // 4, 3 * img_height // 4
            else:  # Others - distributed
                center_x = int(img_width * (0.2 + 0.6 * random.random()))
                center_y = int(img_height * (0.2 + 0.6 * random.random()))
            
            # Calculate box dimensions
            box_width = int(img_width * box_size_factor * 0.6)
            box_height = int(img_height * box_size_factor * 0.4)
            
            # Calculate coordinates
            left = max(0, center_x - box_width // 2)
            top = max(0, center_y - box_height // 2)
            right = min(img_width, center_x + box_width // 2)
            bottom = min(img_height, center_y + box_height // 2)
            
            # Draw box
            draw.rectangle([left, top, right, bottom], outline=color, width=4)
            
            # Add label
            label = f"{obj_name}: {confidence:.2f}"
            self._draw_label(draw, label, left, top, color)
            
            # Store box info
            box_info.append({
                'object': obj_name,
                'confidence': confidence,
                'bounding_box': {
                    'left': left,
                    'top': top,
                    'right': right,
                    'bottom': bottom
                },
                'color': color,
                'detection_method': 'CLIP_estimated'
            })
        
        return box_info
    
    def _draw_label(self, draw: ImageDraw.Draw, label: str, x: int, y: int, color: str):
        """Draw label with background"""
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Calculate text size
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position label above the box
        text_x = x
        text_y = max(0, y - text_height - 5)
        
        # Draw background rectangle
        bg_coords = [text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2]
        draw.rectangle(bg_coords, fill=color, outline=color)
        
        # Draw text
        draw.text((text_x, text_y), label, fill='white', font=font)
    
    def create_results_plot(self, results: Dict, save_path: str = None):
        """Create matplotlib visualization of results"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Load and display image
            image = Image.open(results['image_path'])
            ax1.imshow(image)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Display object detection results
            if 'clip_objects' in results:
                objects = list(results['clip_objects'].keys())[:10]  # Top 10
                scores = list(results['clip_objects'].values())[:10]
                
                ax2.barh(objects, scores)
                ax2.set_xlabel('Confidence Score')
                ax2.set_title('Detected Objects (CLIP)')
                ax2.set_xlim(0, 1)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
