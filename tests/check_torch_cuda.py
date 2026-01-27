import torch
import sys

print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")

# 1. Check CUDA Availability
if torch.cuda.is_available():
    print(f"CUDA Available: YES")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Current CUDA Version (Torch): {torch.version.cuda}")
else:
    print("CUDA Available: NO (CRITICAL FAILURE)")
    sys.exit(1)

# 2. Test Tensor Allocation (The "Smoke Test")
try:
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print("GPU Tensor Operations: SUCCESS (Matrix multiplication worked)")
except Exception as e:
    print(f"GPU Tensor Operations: FAILED with error: {e}")
    sys.exit(1)

# 3. Test Genesis Import
try:
    import genesis as gs
    print("Genesis Import: SUCCESS")
except ImportError as e:
    print(f"Genesis Import: FAILED with error: {e}")
    sys.exit(1)

print("\nResult: Your Environment is SAFE to proceed.")