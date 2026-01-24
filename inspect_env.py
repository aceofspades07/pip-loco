import os

# Path to the environment file
file_path = "genesis_lr/legged_gym/envs/go2/go2.py"

print(f"--- Scanning {file_path} ---")

if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip().startswith("class "):
                print(f"FOUND CLASS: {line.strip()}")
else:
    print("❌ File not found! Check your path.")