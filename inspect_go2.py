import os

# Path to the file that is causing the error
file_path = "genesis_lr/legged_gym/envs/go2/go2_config.py"

print(f"--- Scanning {file_path} ---")

if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip().startswith("class "):
                print(f"FOUND CLASS: {line.strip()}")
else:
    print("❌ File not found! Check your path.")