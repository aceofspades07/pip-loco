import torch
from velocity_estimator import VelocityEstimator

def test_model():
    print("Testing VelocityEstimator...")
    
    # 1. Define Shapes
    batch_size = 32
    history_len = 50
    input_dim = 48
    
    # 2. Instantiate
    model = VelocityEstimator(input_dim, history_len)
    print("Model initialized.")

    # 3. Create Dummy Data (RL Format: Batch, Time, Channels)
    x = torch.randn(batch_size, history_len, input_dim)
    print(f"\n--- INPUT INFO ---")
    print(f"Input Shape: {x.shape}")
    print("Input Data Sample (Batch 0, First 2 Steps, First 4 Sensors):")
    print(x[0, :2, :4]) # Printing a small slice to avoid flooding console

    # 4. Forward Pass
    out = model(x)

    print(f"\n--- OUTPUT INFO ---")
    print(f"Output Shape: {out.shape}")
    print("Output Data Sample (Batch 0 - Est Velocity [vx, vy, vz]):")
    print(out[0]) # Printing the full velocity vector for the first robot

    # 5. Check
    if out.shape == (batch_size, 3):
        print("\nSUCCESS: Output shape is correct (Batch, 3).")
    else:
        print("\nFAILURE: Wrong output shape.")

if __name__ == "__main__":
    test_model()