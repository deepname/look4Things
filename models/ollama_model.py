import ollama
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class OllamaModel:
    """Ollama model for advanced reasoning"""
    
    def __init__(self):
        self.ollama_client = ollama.Client()
    
    def analyze(self, image_description: str, objects_found: Dict[str, float]) -> str:
        """
        Use Ollama for advanced reasoning about identified objects
        
        Args:
            image_description: BLIP-generated description
            objects_found: CLIP identification results
            
        Returns:
            Detailed analysis string
        """
        try:
            # Prepare prompt
            objects_list = ", ".join([f"{obj} ({score:.2f})" for obj, score in objects_found.items()])
            
            prompt = f"""
            Analyze this image based on the following information:
            
            Image Description: {image_description}
            Detected Objects: {objects_list}
            
            Please provide a brief analysis covering:
            1. Most likely objects present
            2. Scene context and setting
            3. Potential use cases or applications
            4. Any notable features or characteristics
            
            Keep the response concise but informative.
            """
            
            response = self.ollama_client.generate(
                model='llama2',  # You can change this to your preferred model
                prompt=prompt
            )
            
            return response['response']
            
        except Exception as e:
            logger.error(f"Ollama analysis failed: {e}")
            return "Advanced analysis not available"
