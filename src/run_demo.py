from data_generator import generate_sample_data
from train import ProductDescriptionGenerator
import torch

def main():
    try:
        print("Checking system setup...")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        print("\nGenerating sample automotive product data...")
        df = generate_sample_data(1000)  # Generate 20 sample products
        
        print("\nSample of generated data:")
        print(df[['name', 'category', 'price']].head())
        
        print("\nInitializing product description generator...")
        generator = ProductDescriptionGenerator()
        
        print("\nPreparing training data...")
        training_texts = generator.prepare_data(df)
        
        print("\nTraining model (this may take a few minutes)...")
        print("Training on device:", "cuda" if torch.cuda.is_available() else "cpu")
        generator.train(training_texts, num_epochs=1)
        
        print("\nGenerating sample description...")
        sample_product = df.iloc[0].to_dict()
        
        print("\nOriginal Description:")
        print(sample_product['original_description'])
        
        print("\nGenerated Enhanced Description:")
        enhanced_description = generator.generate_description(sample_product)
        print(enhanced_description)

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("\nFull error traceback:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
