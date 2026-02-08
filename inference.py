"""
Inference script for wafer defect detection.
Loads ONNX model and classifies wafer images interactively.
"""

import argparse
import sys
from pathlib import Path
import time

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    print("Please install onnxruntime: pip install onnxruntime")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich import print as rprint
    import pyfiglet
except ImportError:
    print("Please install rich and pyfiglet: pip install rich pyfiglet")
    sys.exit(1)


# Initialize Rich Console
console = Console()

# Class names
CLASSES = ['none', 'center', 'donut', 'edge_loc', 'edge_ring', 'loc', 'scratch', 'random']


class WaferClassifier:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.session = None
        self.load_model()

    def load_model(self):
        """Load ONNX model."""
        if not self.model_path.exists():
            console.print(f"[bold red]Error: Model not found at {self.model_path}[/bold red]")
            sys.exit(1)
        
        with console.status(f"[bold green]Loading model: {self.model_path}...[/bold green]"):
            try:
                self.session = ort.InferenceSession(str(self.model_path))
                # Warmup
                dummy_input = np.zeros((1, 1, 64, 64), dtype=np.float32)
                self.session.run(None, {self.session.get_inputs()[0].name: dummy_input})
            except Exception as e:
                console.print(f"[bold red]Failed to load model: {e}[/bold red]")
                sys.exit(1)
        
        console.print(f"[bold green]Model loaded successfully![/bold green]")

    def preprocess_image(self, image_path, size=64):
        """Load and preprocess image for inference."""
        try:
            img = Image.open(image_path).convert('L')  # Grayscale
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            
            # Convert to numpy and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = (img_array - 0.5) / 0.5  # Normalize to [-1, 1]
            
            # Add batch and channel dimensions: (1, 1, H, W)
            img_array = img_array.reshape(1, 1, size, size)
            
            return img_array
        except Exception as e:
            console.print(f"[bold red]Error processing image: {e}[/bold red]")
            return None

    def predict(self, image_array):
        """Run inference on preprocessed image."""
        if self.session is None:
            return None

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: image_array})
        
        # Softmax
        logits = outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        # Get prediction
        pred_idx = np.argmax(probs)
        pred_class = CLASSES[pred_idx]
        confidence = probs[pred_idx]
        
        return pred_class, confidence, probs

    def display_results(self, image_path, pred_class, confidence, probs, top_k=3):
        """Display results in a rich table."""
        # Main Result Panel
        console.print()
        console.print(Panel(f"[bold cyan]Prediction:[/bold cyan] [bold yellow]{pred_class.upper()}[/bold yellow]  |  [bold cyan]Confidence:[/bold cyan] [bold green]{confidence*100:.2f}%[/bold green]", title=f"Analysis: {Path(image_path).name}", expand=False))
        console.print()

        # Detailed Table
        table = Table(title="Top Predictions", show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Class", style="cyan", width=20)
        table.add_column("Confidence", justify="right")
        table.add_column("Bar", width=30)

        top_indices = np.argsort(probs)[::-1][:top_k]
        
        for i, idx in enumerate(top_indices):
            class_name = CLASSES[idx]
            conf = probs[idx]
            bar_len = int(conf * 20)
            bar = "█" * bar_len
            
            # Highlight top prediction
            if i == 0:
                table.add_row(
                    f"[bold]{i+1}[/bold]", 
                    f"[bold yellow]{class_name}[/bold yellow]", 
                    f"[bold green]{conf*100:.2f}%[/bold green]",
                    f"[green]{bar}[/green]"
                )
            else:
                table.add_row(
                    str(i+1), 
                    class_name, 
                    f"{conf*100:.2f}%",
                    f"[white]{bar}[/white]"
                )

        console.print(table)


def print_logo():
    """Print large ASCII logo."""
    console.print()
    f = pyfiglet.Figlet(font='slant')
    console.print(f"[bold blue]{f.renderText('SemiDiff')}[/bold blue]")
    console.print("[bold white]AI–Based Defect Classification System for Semiconductor WaferDie Images[/bold white]", justify="center")
    console.print("[dim]made by Team Ascension | V1.1[/dim]", justify="center")
    console.print()

def main():
    parser = argparse.ArgumentParser(description='Wafer defect inference')
    parser.add_argument('--model', type=str, default='models/model.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Show top-k predictions')
    
    args = parser.parse_args()
    
    # Show Logo
    print_logo()
    
    # Initialize Classifier
    classifier = WaferClassifier(args.model)
    
    # Interactive Loop
    while True:
        console.print("\n[bold]Enter image path[/bold] (or [red]'q'[/red] to quit):")
        user_input = Prompt.ask(">>>")
        
        if user_input.lower() in ('q', 'quit', 'exit'):
            console.print("[yellow]Exiting...[/yellow]")
            break
            
        # Strip quotes if user dragged and dropped file
        image_path = user_input.strip('"\'')
        
        if not image_path:
            continue
            
        path_obj = Path(image_path)
        if not path_obj.exists():
            console.print(f"[bold red]Error: File not found: {image_path}[/bold red]")
            continue
            
        if not path_obj.is_file():
             console.print(f"[bold red]Error: Not a file: {image_path}[/bold red]")
             continue

        # Process
        img_array = classifier.preprocess_image(image_path)
        if img_array is not None:
            result = classifier.predict(img_array)
            if result:
                pred_class, confidence, probs = result
                classifier.display_results(image_path, pred_class, confidence, probs, top_k=args.top_k)

if __name__ == "__main__":
    main()
