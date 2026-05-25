import pandas as pd
import numpy as np
import os

def augment_dataset(input_csv, output_csv, num_augmentations=3, noise_level=0.03):
    """
    Augments the dataset by creating copies and adding random Gaussian noise.
    
    Args:
        input_csv: Path to original CSV file.
        output_csv: Path to save combined (original + augmented) CSV file.
        num_augmentations: Number of times to duplicate and add noise to the dataset.
        noise_level: The scale of noise relative to the standard deviation of each column.
    """
    print(f"Loading '{input_csv}'...")
    df = pd.read_csv(input_csv)
    
    augmented_dfs = [df] # Start with original dataset
    
    # Calculate standard deviation for each column to scale the noise appropriately
    stds = df.std()
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Generating {num_augmentations} augmented copies with normally distributed noise (scale={noise_level}*std)...")
    
    for i in range(num_augmentations):
        # Create a copy
        df_aug = df.copy()
        
        # Add noise to each numerical column
        for col in df.columns:
            # Generate Gaussian noise
            noise = np.random.normal(0, stds[col] * noise_level, size=len(df_aug))
            df_aug[col] = df_aug[col] + noise
            
            # ── Apply agronomically realistic bounds ──
            
            # CATEGORY 1: CANNOT be zero (physical/biological impossibility)
            if col == 'Soil_pH':
                # pH scale: agricultural soil is always 3.5–9.5
                df_aug[col] = df_aug[col].clip(lower=3.5, upper=9.5)
            elif col == 'Sunlight_Hours':
                # A crop cycle needs sunlight to produce yield; min ~1 hr
                df_aug[col] = df_aug[col].clip(lower=1.0, upper=16.0)
            
            # CATEGORY 2: SHOULD NOT be zero (sensor glitch / wilting point)
            elif col == 'Soil_Moisture':
                # Below ~5% is past permanent wilting point for most crops
                df_aug[col] = df_aug[col].clip(lower=5.0, upper=100.0)
            elif col == 'Temperature':
                # Maize needs >5°C to germinate; season avg can't be 0
                df_aug[col] = df_aug[col].clip(lower=5.0, upper=45.0)
            elif col == 'Humidity':
                # 0% RH is virtually impossible in agricultural zones
                df_aug[col] = df_aug[col].clip(lower=10.0, upper=100.0)
            
            # CATEGORY 3: CAN be zero (episodic / depletion scenarios)
            elif col == 'Rainfall':
                # 0mm is perfectly valid (dry spell or irrigated field)
                df_aug[col] = df_aug[col].clip(lower=0.0)
            elif col in ['N', 'P', 'K']:
                # Depleted soil is rare but possible; keep a tiny floor
                df_aug[col] = df_aug[col].clip(lower=1.0)
            elif col == 'Crop_Yield_ton_per_hectare':
                # Yield must be positive
                df_aug[col] = df_aug[col].clip(lower=0.1)
                
            # Rounding for cleanliness
            if col in ['N', 'P', 'K']:
                df_aug[col] = df_aug[col].round(0).astype(int)
            else:
                df_aug[col] = df_aug[col].round(2)
                
        augmented_dfs.append(df_aug)
        
    # Combine all
    final_df = pd.concat(augmented_dfs, ignore_index=True)
    
    # Shuffle the dataset
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final augmented dataset shape: {final_df.shape}")
    
    # Save to CSV
    final_df.to_csv(output_csv, index=False)
    print(f"Saved augmented data to '{output_csv}'")

if __name__ == "__main__":
    input_file = "yield-predict.csv"
    output_file = "yield-predict-augmented.csv"
    
    if os.path.exists(input_file):
        augment_dataset(input_file, output_file, num_augmentations=4, noise_level=0.05)
    else:
        print(f"Error: Could not find '{input_file}'. Please ensure you are in the correct directory.")
