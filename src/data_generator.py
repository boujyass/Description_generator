import pandas as pd
import random
import os

def generate_sample_data(num_samples=100):
    # Product categories and their attributes
    categories = {
        'Brakes': {
            'products': ['Brake Pads', 'Brake Rotors', 'Brake Calipers', 'Brake Lines'],
            'materials': ['Ceramic', 'Semi-Metallic', 'Organic', 'Carbon Fiber'],
            'positions': ['Front', 'Rear', 'All Axle'],
            'specs': ['High Performance', 'Heavy Duty', 'Standard Duty', 'Sport']
        },
        'Filters': {
            'products': ['Oil Filter', 'Air Filter', 'Fuel Filter', 'Cabin Filter'],
            'materials': ['Synthetic', 'Paper', 'Cotton Gauze', 'Foam'],
            'sizes': ['Standard', 'Oversized', 'Compact', 'Extended Life'],
            'grades': ['Premium', 'OEM', 'Performance', 'Economy']
        },
        'Engine': {
            'products': ['Spark Plugs', 'Pistons', 'Gaskets', 'Timing Belts'],
            'materials': ['Iridium', 'Platinum', 'Copper', 'Aluminum'],
            'specs': ['Stock', 'Performance', 'Racing', 'Heavy Duty'],
            'grades': ['OEM', 'Aftermarket', 'Performance', 'Premium']
        }
    }

    data = []
    for _ in range(num_samples):
        # Select random category
        category = random.choice(list(categories.keys()))
        cat_data = categories[category]
        
        # Generate product data
        product = random.choice(cat_data['products'])
        material = random.choice(cat_data['materials'])
        spec = random.choice(cat_data.get('specs', ['Standard', 'Premium', 'Basic']))
        
        # Generate metadata
        metadata = {
            'material': material,
            'specification': spec,
            'compatibility': random.choice(['Universal', 'Vehicle Specific', 'Multiple Models']),
            'warranty': random.choice(['1 year', '2 years', '3 years', 'Lifetime'])
        }
        
        # Generate descriptions
        original_desc = f"Standard {product.lower()} for automotive applications. Made with {material.lower()} material."
        
        data.append({
            'product_id': f'P{random.randint(1000, 9999)}',
            'name': f'{spec} {material} {product}',
            'category': category,
            'metadata': metadata,  # keep as dictionary for easier handling
            'original_description': original_desc,
            'price': round(random.uniform(20.0, 500.0), 2),
            'stock': random.randint(0, 100)
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_data(100)
    # Save to CSV
    df.to_csv('sample_products.csv', index=False)
    print("Sample data generated successfully!")
