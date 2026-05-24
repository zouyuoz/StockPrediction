import jax
import torch
import sys

def check_env():
    print(f"Python version: {sys.version}")
    print("\n--- Checking JAX ---")
    try:
        print(f"JAX version: {jax.__version__}")
        devices = jax.devices()
        print(f"JAX devices: {devices}")
        if any(d.device_kind.lower() == 'gpu' for d in devices):
            print("✅ JAX can see the GPU.")
        else:
            print("❌ JAX cannot see the GPU.")
    except Exception as e:
        print(f"❌ JAX error: {e}")

    print("\n--- Checking PyTorch ---")
    try:
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ PyTorch can see the GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version used by PyTorch: {torch.version.cuda}")
        else:
            print("❌ PyTorch cannot see the GPU.")
    except Exception as e:
        print(f"❌ PyTorch error: {e}")

if __name__ == "__main__":
    check_env()
