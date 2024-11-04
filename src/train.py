import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset as TorchDataset
import os

class ProductDescriptionDataset(TorchDataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove the batch dimension
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item['labels'] = item['input_ids'].clone()
        
        return item

class ProductDescriptionGenerator:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def prepare_data(self, df):
        """
        Prepare training data from DataFrame
        """
        training_texts = []
        
        for _, row in df.iterrows():
            metadata = row['metadata']  # Access directly as dictionary
            
            # Create the combined text
            text = f"""Input:
Product: {row['name']}
Category: {row['category']}
Metadata: {metadata}
Original Description: {row['original_description']}
Price: ${row['price']}
Output:
Introducing the {row['name']}, a premium {row['category'].lower()} solution 
designed for optimal performance and reliability. This {metadata['material'].lower()}-based product 
features {metadata['specification']} specifications, making it perfect for professional automotive applications. 
{row['original_description']} Backed by a {metadata['warranty']} warranty and engineered for 
{metadata['compatibility'].lower()} compatibility. 
Available now for ${row['price']:.2f}.
END"""
            
            training_texts.append(text)
        
        print(f"Prepared {len(training_texts)} training examples")
        return training_texts

    def train(self, training_texts, output_dir="./model", num_epochs=3):
        os.makedirs(output_dir, exist_ok=True)

        print("Preparing dataset...")
        train_dataset = ProductDescriptionDataset(training_texts, self.tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,
            save_steps=100,
            save_total_limit=2,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_first_step=True,
            use_cpu=not torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        print("Starting training...")
        trainer.train()
        
        print("Saving model...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("Training completed!")

    def generate_description(self, product_info, max_length=200):
        metadata = product_info['metadata']
        
        input_text = f"""Input:
Product: {product_info['name']}
Category: {product_info['category']}
Metadata: {metadata}
Original Description: {product_info['original_description']}
Price: ${product_info['price']}
Output:"""

        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
        
        print("Generating description...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text.split("Output:")[-1].strip()

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"Model loaded from {path}")
