#!/usr/bin/env python3
"""
Command-line interface for the AI Object Identifier
"""

import argparse
import json
import sys
import os
from pathlib import Path
from object_identifier import ObjectIdentifier
import logging

def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def validate_image_path(path):
    """Validate that the image path exists and is a valid image file"""
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"Image file does not exist: {path}")
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    if Path(path).suffix.lower() not in valid_extensions:
        raise argparse.ArgumentTypeError(f"Invalid image format. Supported: {valid_extensions}")
    
    return path

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="🔍 AI Object Identifier - Identify objects in images using multiple AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.jpg                           # Basic identification
  %(prog)s image.jpg --objects "car,person"   # Look for specific objects
  %(prog)s image.jpg --output results.json    # Save results to file
  %(prog)s image.jpg --visualize              # Show visualization
  %(prog)s --web                              # Launch web interface
        """
    )
    
    # Main arguments
    parser.add_argument(
        'image',
        nargs='?',
        type=validate_image_path,
        help='Path to the image file to analyze'
    )
    
    # Object specification
    parser.add_argument(
        '--objects', '-o',
        type=str,
        help='Comma-separated list of objects to look for (e.g., "car,person,dog")'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-O',
        type=str,
        help='Save results to JSON file'
    )
    
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Show visualization of results'
    )
    
    parser.add_argument(
        '--save-viz',
        type=str,
        help='Save visualization to image file'
    )
    
    # Interface options
    parser.add_argument(
        '--web', '-w',
        action='store_true',
        help='Launch web interface instead of CLI'
    )
    
    parser.add_argument(
        '--batch',
        type=str,
        help='Process multiple images from directory'
    )
    
    # Model options
    parser.add_argument(
        '--no-clip',
        action='store_true',
        help='Disable CLIP model'
    )
    
    parser.add_argument(
        '--no-blip',
        action='store_true',
        help='Disable BLIP model'
    )
    
    parser.add_argument(
        '--no-ollama',
        action='store_true',
        help='Disable Ollama analysis'
    )
    
    # System options
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Device to use for inference (default: auto)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='AI Object Identifier v1.0.0'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Handle web interface
    if args.web:
        logger.info("Launching web interface...")
        try:
            from web_interface import ObjectIdentifierInterface
            app = ObjectIdentifierInterface()
            app.launch(share=True)
        except ImportError as e:
            logger.error(f"Web interface dependencies missing: {e}")
            sys.exit(1)
        return
    
    # Validate required arguments for CLI mode
    if not args.image and not args.batch:
        parser.error("Either provide an image file or use --web for web interface")
    
    # Initialize object identifier
    logger.info("Initializing AI models...")
    try:
        device = None if args.device == 'auto' else args.device
        identifier = ObjectIdentifier(device=device)
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        sys.exit(1)
    
    # Process batch directory
    if args.batch:
        process_batch(identifier, args, logger)
        return
    
    # Process single image
    process_single_image(identifier, args, logger)

def process_single_image(identifier, args, logger):
    """Process a single image"""
    logger.info(f"Analyzing image: {args.image}")
    
    # Prepare candidate objects
    candidate_objects = None
    if args.objects:
        candidate_objects = [obj.strip() for obj in args.objects.split(',') if obj.strip()]
        logger.info(f"Looking for specific objects: {candidate_objects}")
    
    try:
        # Perform identification
        results = identifier.identify_comprehensive(args.image, candidate_objects)
        
        # Handle errors
        if 'error' in results:
            logger.error(f"Analysis failed: {results['error']}")
            sys.exit(1)
        
        # Display results
        display_results(results, args, logger)
        
        # Always save results to result folder with bounding boxes
        logger.info("Saving results to 'result' folder...")
        saved_files = identifier.save_results_to_folder(results, "result")
        
        if saved_files.get('json_path'):
            logger.info(f"Results saved to: {saved_files['json_path']}")
        if saved_files.get('image_with_boxes_path'):
            logger.info(f"Image with bounding boxes saved to: {saved_files['image_with_boxes_path']}")
        
        # Save results if requested (additional output)
        if args.output:
            save_results(results, args.output, logger)
        
        # Show/save visualization if requested
        if args.visualize or args.save_viz:
            handle_visualization(identifier, results, args, logger)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

def process_batch(identifier, args, logger):
    """Process multiple images from a directory"""
    batch_dir = Path(args.batch)
    if not batch_dir.exists() or not batch_dir.is_dir():
        logger.error(f"Batch directory does not exist: {args.batch}")
        sys.exit(1)
    
    # Find image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in batch_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.warning(f"No image files found in {args.batch}")
        return
    
    logger.info(f"Processing {len(image_files)} images...")
    
    batch_results = {}
    for i, image_file in enumerate(image_files, 1):
        logger.info(f"Processing {i}/{len(image_files)}: {image_file.name}")
        
        try:
            candidate_objects = None
            if args.objects:
                candidate_objects = [obj.strip() for obj in args.objects.split(',')]
            
            results = identifier.identify_comprehensive(str(image_file), candidate_objects)
            batch_results[str(image_file)] = results
            
            # Quick summary
            if 'clip_objects' in results and results['clip_objects']:
                top_object = list(results['clip_objects'].keys())[0]
                top_score = list(results['clip_objects'].values())[0]
                logger.info(f"  → Top object: {top_object} ({top_score:.3f})")
            
        except Exception as e:
            logger.error(f"  → Failed: {e}")
            batch_results[str(image_file)] = {'error': str(e)}
    
    # Save batch results
    if args.output:
        save_results(batch_results, args.output, logger)
    
    logger.info(f"Batch processing completed! Processed {len(image_files)} images.")

def display_results(results, args, logger):
    """Display results in a formatted way"""
    print("\n" + "="*60)
    print("🔍 OBJECT IDENTIFICATION RESULTS")
    print("="*60)
    
    # Image caption
    if 'caption' in results and results['caption']:
        print(f"\n📝 Image Description:")
        print(f"   {results['caption']}")
    
    # Top detected objects
    if 'clip_objects' in results and results['clip_objects']:
        print(f"\n🎯 Top Detected Objects:")
        for i, (obj, score) in enumerate(list(results['clip_objects'].items())[:10], 1):
            confidence_bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"   {i:2d}. {obj.title():<15} {score:.3f} [{confidence_bar}]")
    
    # Advanced analysis
    if 'advanced_analysis' in results and results['advanced_analysis']:
        print(f"\n🧠 AI Analysis:")
        # Format the analysis text with proper indentation
        analysis_lines = results['advanced_analysis'].split('\n')
        for line in analysis_lines:
            if line.strip():
                print(f"   {line}")
    
    # Technical info
    print(f"\n⚙️ Technical Information:")
    print(f"   Segmentation Available: {'Yes' if results.get('segmentation_available') else 'No'}")
    print(f"   Processing Device: {results.get('timestamp', 'Unknown')}")
    
    print("\n" + "="*60)

def save_results(results, output_path, logger):
    """Save results to JSON file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

def handle_visualization(identifier, results, args, logger):
    """Handle visualization display/saving"""
    try:
        if args.visualize:
            identifier.visualize_results(results)
        
        if args.save_viz:
            identifier.visualize_results(results, args.save_viz)
            logger.info(f"Visualization saved to: {args.save_viz}")
            
    except Exception as e:
        logger.error(f"Visualization failed: {e}")

if __name__ == "__main__":
    main()
