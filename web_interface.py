import gradio as gr
import json
import os
from object_identifier import ObjectIdentifier
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import base64

class ObjectIdentifierInterface:
    """Web interface for the Object Identifier using Gradio"""
    
    def __init__(self):
        """Initialize the interface and object identifier"""
        self.identifier = None
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize the ObjectIdentifier with error handling"""
        try:
            self.identifier = ObjectIdentifier()
            return "✅ Models initialized successfully!"
        except Exception as e:
            return f"❌ Error initializing models: {str(e)}"
    
    def identify_objects(self, image, custom_objects_text, use_default_objects):
        """
        Main function to identify objects in uploaded image
        
        Args:
            image: Uploaded image from Gradio
            custom_objects_text: Comma-separated list of objects to look for
            use_default_objects: Whether to use default object list
            
        Returns:
            Tuple of (results_json, visualization_image, status_message)
        """
        if self.identifier is None:
            return "Error: Models not initialized", None, "❌ Models not loaded"
        
        if image is None:
            return "No image provided", None, "❌ Please upload an image"
        
        try:
            # Save uploaded image temporarily
            temp_path = "temp_image.jpg"
            image.save(temp_path)
            
            # Prepare candidate objects
            candidate_objects = None
            if not use_default_objects and custom_objects_text.strip():
                candidate_objects = [obj.strip() for obj in custom_objects_text.split(',') if obj.strip()]
            
            # Perform identification
            results = self.identifier.identify_comprehensive(temp_path, candidate_objects)
            
            # Create visualization
            viz_image = self.create_visualization(results, image)
            
            # Format results for display
            formatted_results = self.format_results(results)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return formatted_results, viz_image, "✅ Analysis completed successfully!"
            
        except Exception as e:
            return f"Error during analysis: {str(e)}", None, f"❌ Analysis failed: {str(e)}"
    
    def create_visualization(self, results, original_image):
        """Create a visualization of the results"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Display original image
            ax1.imshow(original_image)
            ax1.set_title('Original Image', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # Display object detection results
            if 'clip_objects' in results and results['clip_objects']:
                objects = list(results['clip_objects'].keys())[:10]  # Top 10
                scores = list(results['clip_objects'].values())[:10]
                
                # Create horizontal bar chart
                y_pos = np.arange(len(objects))
                bars = ax2.barh(y_pos, scores, color='skyblue', edgecolor='navy', alpha=0.7)
                
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(objects)
                ax2.set_xlabel('Confidence Score', fontweight='bold')
                ax2.set_title('Top Detected Objects (CLIP)', fontsize=14, fontweight='bold')
                ax2.set_xlim(0, max(scores) * 1.1 if scores else 1)
                
                # Add value labels on bars
                for i, (bar, score) in enumerate(zip(bars, scores)):
                    ax2.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{score:.3f}', va='center', fontsize=10)
            else:
                ax2.text(0.5, 0.5, 'No objects detected', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Object Detection Results')
            
            plt.tight_layout()
            
            # Convert to image for Gradio
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            viz_image = Image.open(buf)
            plt.close()
            
            return viz_image
            
        except Exception as e:
            print(f"Visualization error: {e}")
            return None
    
    def format_results(self, results):
        """Format results for better display"""
        if 'error' in results:
            return f"Error: {results['error']}"
        
        formatted = "# 🔍 Object Identification Results\n\n"
        
        # Image caption
        if 'caption' in results and results['caption']:
            formatted += f"## 📝 Image Description\n{results['caption']}\n\n"
        
        # Top detected objects
        if 'clip_objects' in results and results['clip_objects']:
            formatted += "## 🎯 Detected Objects (Top 10)\n"
            for i, (obj, score) in enumerate(list(results['clip_objects'].items())[:10], 1):
                confidence_bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                formatted += f"{i:2d}. **{obj.title()}** - {score:.3f} `{confidence_bar}`\n"
            formatted += "\n"
        
        # Advanced analysis
        if 'advanced_analysis' in results and results['advanced_analysis']:
            formatted += f"## 🧠 AI Analysis\n{results['advanced_analysis']}\n\n"
        
        # Technical info
        formatted += "## ⚙️ Technical Information\n"
        formatted += f"- **Segmentation Available**: {'Yes' if results.get('segmentation_available') else 'No'}\n"
        formatted += f"- **Processing Device**: {results.get('timestamp', 'Unknown')}\n"
        
        return formatted
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            max-width: 1200px !important;
        }
        .output-markdown {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        """
        
        with gr.Blocks(css=css, title="🔍 AI Object Identifier", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("""
            # 🔍 AI Object Identifier
            
            Upload an image and let multiple AI models identify objects for you!
            
            **Features:**
            - 🖼️ **BLIP**: Generates natural language descriptions
            - 🎯 **CLIP**: Identifies specific objects with confidence scores  
            - 🧠 **Ollama**: Provides advanced AI analysis
            - ✂️ **SAM**: Object segmentation (when available)
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Input section
                    gr.Markdown("## 📤 Upload & Configure")
                    
                    image_input = gr.Image(
                        type="pil", 
                        label="Upload Image",
                        height=300
                    )
                    
                    use_default = gr.Checkbox(
                        label="Use default object categories", 
                        value=True,
                        info="Includes: person, car, dog, cat, etc."
                    )
                    
                    custom_objects = gr.Textbox(
                        label="Custom objects to look for (comma-separated)",
                        placeholder="e.g., laptop, coffee cup, smartphone, book",
                        lines=2,
                        interactive=True
                    )
                    
                    analyze_btn = gr.Button(
                        "🔍 Analyze Image", 
                        variant="primary",
                        size="lg"
                    )
                    
                    status_output = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=1
                    )
                
                with gr.Column(scale=2):
                    # Output section
                    gr.Markdown("## 📊 Results")
                    
                    with gr.Tab("📋 Analysis Report"):
                        results_output = gr.Markdown(
                            label="Detailed Results",
                            value="Upload an image and click 'Analyze Image' to see results here."
                        )
                    
                    with gr.Tab("📈 Visualization"):
                        viz_output = gr.Image(
                            label="Results Visualization",
                            type="pil"
                        )
            
            # Examples section
            gr.Markdown("## 🖼️ Example Images")
            gr.Markdown("Try these example scenarios:")
            
            example_scenarios = [
                "Upload a photo of your workspace to identify office items",
                "Take a picture of your meal to identify food items", 
                "Upload a street scene to detect vehicles and people",
                "Try a nature photo to identify animals and plants"
            ]
            
            for scenario in example_scenarios:
                gr.Markdown(f"• {scenario}")
            
            # Connect the analyze button
            analyze_btn.click(
                fn=self.identify_objects,
                inputs=[image_input, custom_objects, use_default],
                outputs=[results_output, viz_output, status_output]
            )
            
            # Footer
            gr.Markdown("""
            ---
            **Powered by**: CLIP, BLIP, SAM, Ollama | **Built with**: Python & Gradio
            """)
        
        return interface
    
    def launch(self, share=False, debug=False):
        """Launch the Gradio interface"""
        interface = self.create_interface()
        
        print("🚀 Starting Object Identifier Web Interface...")
        print("📊 Loading AI models (this may take a moment)...")
        
        interface.launch(
            share=share,
            debug=debug,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )

if __name__ == "__main__":
    # Create and launch the interface
    app = ObjectIdentifierInterface()
    
    print("🔍 AI Object Identifier")
    print("=" * 50)
    
    # Launch with sharing enabled for easy access
    app.launch(share=True, debug=True)
