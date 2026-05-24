import torch
import numpy as np
import timesfm
import os

# Set precision for potential speedup
torch.set_float32_matmul_precision("high")

def test_inference():
    print("--- Initializing TimesFM 2.5 (Torch) ---")
    try:
        # Load the model. This will download weights from Hugging Face if not present.
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        
        print("--- Compiling model with ForecastConfig ---")
        model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=256,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        
        print("--- Running Dummy Forecast ---")
        # Two dummy inputs: a linear sequence and a sine wave
        inputs = [
            np.linspace(0, 1, 100),
            np.sin(np.linspace(0, 20, 67)),
        ]
        
        point_forecast, quantile_forecast = model.forecast(
            horizon=12,
            inputs=inputs,
        )
        
        print(f"Point forecast shape: {point_forecast.shape}")
        print(f"Quantile forecast shape: {quantile_forecast.shape}")
        
        if point_forecast.shape == (2, 12):
            print("✅ TimesFM inference successful!")
        else:
            print(f"⚠️ Unexpected output shape: {point_forecast.shape}")

    except Exception as e:
        print(f"❌ Error during TimesFM test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()
