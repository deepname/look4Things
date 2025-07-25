import os
import json
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class FileManager:
    """Handles file operations and result saving"""
    
    @staticmethod
    def save_results_to_folder(results: Dict, output_dir: str = "result") -> Dict[str, str]:
        """
        Save comprehensive results including JSON and image with bounding boxes
        
        Args:
            results: Results from identify_comprehensive
            output_dir: Directory to save all results
            
        Returns:
            Dictionary with paths to saved files
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Get base filename
            base_name = os.path.splitext(os.path.basename(results['image_path']))[0]
            
            # Save JSON results
            json_path = os.path.join(output_dir, f"{base_name}_results.json")
            
            # Save enhanced JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            saved_files = {
                'json_path': json_path,
                'image_with_boxes_path': results.get('output_image_path')
            }
            
            logger.info(f"Results saved to folder: {output_dir}")
            logger.info(f"JSON results: {json_path}")
            if results.get('output_image_path'):
                logger.info(f"Image with boxes: {results['output_image_path']}")
            
            return saved_files
            
        except Exception as e:
            logger.error(f"Failed to save results to folder: {e}")
            return {}
    
    @staticmethod
    def save_json(data: Dict, file_path: str) -> bool:
        """
        Save dictionary to JSON file
        
        Args:
            data: Dictionary to save
            file_path: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"JSON saved to: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON to {file_path}: {e}")
            return False
    
    @staticmethod
    def load_json(file_path: str) -> Dict:
        """
        Load JSON file
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Loaded dictionary or empty dict if failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON from {file_path}: {e}")
            return {}
