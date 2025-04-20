from ml_models import TeamIdentifier
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train the team identification model')
    parser.add_argument('--matches_file', type=str, required=True, help='Path to matches JSON file')
    parser.add_argument('--output_dir', type=str, default='models/team_identifier', help='Directory to save the model')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize and train the model
    print("Initializing team identifier...")
    identifier = TeamIdentifier()
    
    print(f"Training on {args.matches_file}...")
    identifier.train(
        matches_file=args.matches_file,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print(f"Saving model to {args.output_dir}...")
    identifier.save_model(args.output_dir)
    print("Training complete!")

if __name__ == "__main__":
    main() 