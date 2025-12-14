"""
Setup script to download required NLTK data.
Run this before using the evaluation pipeline.
"""

import nltk
import sys

def download_nltk_data():
    """Download required NLTK resources."""
    print("Downloading NLTK data...")
    
    resources = [
        ('punkt_tab', 'punkt_tab'),
        ('punkt', 'punkt'),  # Fallback for older versions
        ('stopwords', 'stopwords')
    ]
    
    for resource_name, resource_id in resources:
        try:
            print(f"Downloading {resource_name}...", end=' ')
            nltk.download(resource_id, quiet=True)
            print("✓")
        except Exception as e:
            print(f"✗ (Error: {e})")
            if resource_name == 'punkt_tab':
                # Try punkt as fallback
                try:
                    print(f"Trying fallback 'punkt'...", end=' ')
                    nltk.download('punkt', quiet=True)
                    print("✓")
                except:
                    print("✗")
    
    print("\nNLTK data download complete!")
    print("You can now run the evaluation pipeline.")

if __name__ == "__main__":
    try:
        download_nltk_data()
    except ImportError:
        print("Error: NLTK is not installed.")
        print("Please install it with: pip install nltk")
        sys.exit(1)

