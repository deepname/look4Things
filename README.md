# 🔍 AI Object Identifier

A comprehensive object identification system using multiple state-of-the-art AI models including CLIP, BLIP, SAM, and Ollama for advanced computer vision and natural language processing.

## ✨ Features

- **🖼️ BLIP Integration**: Generates natural language descriptions of images
- **🎯 CLIP Integration**: Identifies objects with confidence scores using text-image similarity
- **✂️ SAM Integration**: Segment Anything Model for precise object segmentation
- **🧠 Ollama Integration**: Advanced AI reasoning and analysis
- **🌐 Web Interface**: Beautiful Gradio-based web UI
- **💻 CLI Interface**: Powerful command-line tool for batch processing
- **📊 Visualization**: Rich charts and graphs for results analysis
- **🚀 GPU Acceleration**: Automatic CUDA detection and optimization

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd search\ dirt
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download SAM checkpoint** (optional, for segmentation):
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

4. **Install Ollama** (for advanced analysis):
```bash
# Visit https://ollama.ai for installation instructions
# Then pull a model:
ollama pull llama2
```

### Usage

#### 🌐 Web Interface (Recommended)
```bash
python web_interface.py
```
Then open your browser to `http://localhost:7860`

#### 💻 Command Line Interface
```bash
# Basic usage
python cli.py image.jpg

# Look for specific objects
python cli.py image.jpg --objects "car,person,dog"

# Save results and show visualization
python cli.py image.jpg --output results.json --visualize

# Process multiple images
python cli.py --batch /path/to/images/ --output batch_results.json

# Launch web interface from CLI
python cli.py --web
```

#### 🐍 Python API
```python
from object_identifier import ObjectIdentifier

# Initialize
identifier = ObjectIdentifier()

# Analyze image
results = identifier.identify_comprehensive("path/to/image.jpg")

# Print results
print(f"Caption: {results['caption']}")
for obj, score in results['clip_objects'].items():
    print(f"{obj}: {score:.3f}")
```

## 📁 Project Structure

```
search dirt/
├── object_identifier.py    # Main ObjectIdentifier class
├── web_interface.py        # Gradio web interface
├── cli.py                 # Command-line interface
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── examples/             # Example images (add your own)
```

## 🔧 Configuration

### Model Configuration

The system automatically detects available hardware and loads models accordingly:

- **GPU Detection**: Automatically uses CUDA if available
- **Model Fallbacks**: Gracefully handles missing models
- **Memory Optimization**: Efficient model loading and inference

### Customization

You can customize the behavior by modifying the `ObjectIdentifier` class:

```python
# Custom device selection
identifier = ObjectIdentifier(device="cuda")

# Custom object categories
custom_objects = ["laptop", "smartphone", "coffee cup"]
results = identifier.identify_comprehensive("image.jpg", custom_objects)
```

## 🎯 Supported Models

### CLIP (Contrastive Language-Image Pre-training)
- **Purpose**: Text-image similarity matching
- **Model**: ViT-B/32
- **Use Case**: Object identification with confidence scores

### BLIP (Bootstrapping Language-Image Pre-training)
- **Purpose**: Image captioning
- **Model**: Salesforce/blip-image-captioning-base
- **Use Case**: Natural language image descriptions

### SAM (Segment Anything Model)
- **Purpose**: Object segmentation
- **Model**: ViT-H checkpoint
- **Use Case**: Precise object boundaries and masks

### Ollama
- **Purpose**: Advanced reasoning and analysis
- **Model**: Configurable (llama2, codellama, etc.)
- **Use Case**: Contextual understanding and insights

## 📊 Output Format

The system returns comprehensive results in JSON format:

```json
{
  "image_path": "path/to/image.jpg",
  "caption": "A person sitting at a desk with a laptop",
  "clip_objects": {
    "person": 0.892,
    "laptop": 0.756,
    "desk": 0.643,
    "chair": 0.521
  },
  "advanced_analysis": "The image shows a typical office workspace...",
  "segmentation_available": true,
  "timestamp": "GPU_0"
}
```

## 🌐 Web Interface Features

The Gradio web interface provides:

- **📤 Drag & Drop Upload**: Easy image uploading
- **🎛️ Custom Object Search**: Specify objects to look for
- **📊 Interactive Visualizations**: Charts and graphs
- **📱 Mobile Friendly**: Responsive design
- **🔗 Shareable Links**: Share your interface publicly

## 💻 CLI Features

The command-line interface supports:

- **📁 Batch Processing**: Process entire directories
- **💾 JSON Export**: Save results for further analysis
- **🖼️ Visualization**: Generate and save result charts
- **⚙️ Model Control**: Enable/disable specific models
- **📝 Verbose Logging**: Detailed processing information

## 🔧 Advanced Usage

### Batch Processing
```bash
# Process all images in a directory
python cli.py --batch ./images/ --output batch_results.json

# Process with custom objects
python cli.py --batch ./images/ --objects "car,truck,motorcycle" --output vehicle_analysis.json
```

### Custom Analysis
```python
from object_identifier import ObjectIdentifier

identifier = ObjectIdentifier()

# Analyze with custom objects
results = identifier.identify_comprehensive(
    "image.jpg", 
    candidate_objects=["laptop", "mouse", "keyboard", "monitor"]
)

# Generate visualization
identifier.visualize_results(results, save_path="analysis.png")
```

### Integration with Other Tools
```python
# Use with OpenCV
import cv2
from object_identifier import ObjectIdentifier

identifier = ObjectIdentifier()
image = cv2.imread("image.jpg")

# Convert and analyze
from PIL import Image
pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# ... continue with analysis
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   python cli.py image.jpg --device cpu
   ```

2. **Missing SAM Checkpoint**:
   - Download from the official repository
   - Or disable SAM in the code

3. **Ollama Connection Error**:
   - Ensure Ollama is running: `ollama serve`
   - Check if models are installed: `ollama list`

4. **Import Errors**:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

### Performance Tips

- **Use GPU**: Ensure CUDA is properly installed
- **Batch Processing**: Process multiple images at once
- **Model Caching**: Models are cached after first load
- **Memory Management**: Close visualization windows to free memory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI CLIP**: For the amazing text-image understanding
- **Salesforce BLIP**: For high-quality image captioning
- **Meta SAM**: For revolutionary segmentation capabilities
- **Ollama**: For local LLM integration
- **Gradio**: For the beautiful web interface

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Look at the example usage in the code
3. Create an issue in the repository
4. Check the model documentation for specific models

## 🔮 Future Enhancements

- [ ] Real-time video analysis
- [ ] Custom model fine-tuning
- [ ] API endpoint deployment
- [ ] Mobile app integration
- [ ] Cloud deployment options
- [ ] Advanced segmentation features
- [ ] Multi-language support

---

**Made with ❤️ using Python, CLIP, BLIP, SAM, and Ollama**
