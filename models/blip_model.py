import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging

logger = logging.getLogger(__name__)

class BLIPModel:
    """BLIP model for image captioning"""
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.blip_processor = None
        self.blip_model = None
        self._load_model()
    
    def _load_model(self):
        """Load BLIP model"""
        try:
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model.to(self.device)
            logger.info("BLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            self.blip_model = None
    
    def is_available(self) -> bool:
        """Check if BLIP model is available"""
        return self.blip_model is not None
    
    def generate_caption(self, image: Image.Image) -> str:
        """
        Generate image caption using BLIP
        
        Args:
            image: PIL Image
            
        Returns:
            Generated caption string
        """
        if not self.is_available():
            return ""
        
        try:
            # Process image
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
            
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            logger.error(f"BLIP captioning failed: {e}")
            return ""
