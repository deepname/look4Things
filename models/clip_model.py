import torch
import clip
from PIL import Image
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class CLIPModel:
    """CLIP model for image-text similarity"""
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = None
        self.clip_preprocess = None
        self._load_model()
    
    def _load_model(self):
        """Load CLIP model"""
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.clip_model = None
    
    def is_available(self) -> bool:
        """Check if CLIP model is available"""
        return self.clip_model is not None
    
    def identify_objects(self, image: Image.Image, candidate_objects: List[str]) -> Dict[str, float]:
        """
        Use CLIP to identify objects in image based on text descriptions
        
        Args:
            image: PIL Image
            candidate_objects: List of object names to check for
            
        Returns:
            Dictionary with object names and confidence scores
        """
        if not self.is_available():
            return {}
        
        try:
            # Preprocess image
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize text descriptions
            text_inputs = clip.tokenize([f"a photo of a {obj}" for obj in candidate_objects]).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                logits_per_image, logits_per_text = self.clip_model(image_input, text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            # Return results
            results = {obj: float(prob) for obj, prob in zip(candidate_objects, probs)}
            return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"CLIP identification failed: {e}")
            return {}
