import jax
import jax.numpy as jnp
import sys

def check_jax():
    print(f"Python version: {sys.version}")
    print(f"JAX version: {jax.__version__}")
    
    try:
        devices = jax.devices()
        print(f"Available devices: {devices}")
        
        for i, d in enumerate(devices):
            print(f"Device {i}: {d.device_kind} (platform: {d.platform})")
        
        # Try a simple computation on GPU
        print("\n--- Testing JAX Computation ---")
        x = jnp.ones((1000, 1000))
        y = jnp.dot(x, x)
        # Force execution
        y.block_until_ready()
        print("✅ JAX computation successful on device!")
        
        # Check if it's actually using the GPU
        if any(d.platform == 'gpu' for d in devices):
            print("✅ JAX is confirmed to be using the GPU (platform: gpu).")
        else:
            print("⚠️ JAX is running, but no 'gpu' platform found in devices. It might be falling back to CPU.")
            
    except Exception as e:
        print(f"❌ JAX error during computation: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True

if __name__ == "__main__":
    check_jax()
