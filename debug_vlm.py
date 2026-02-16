
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, GenerationConfig
import traceback
import sys

def debug_load():
    model_id = "vikhyatk/moondream2"
    revision = "2025-01-09"
    device = "cpu"
    
    print(f"DEBUG: Starting Load on {device}...")
    
    # Apply monkey patch exactly as in vlm.py
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        @property
        def all_tied_weights_keys(self):
            return {}
        PreTrainedModel.all_tied_weights_keys = all_tied_weights_keys
        print("DEBUG: Applied all_tied_weights_keys patch")

    try:
        print("DEBUG: Calling AutoModelForCausalLM.from_pretrained...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=torch.float32,
        ).to(device)
        
        print("DEBUG: Model loaded. Setting generation config...")
        try:
            model.generation_config = GenerationConfig.from_pretrained(model_id, trust_remote_code=True)
            print("DEBUG: Loaded remote GenerationConfig")
        except Exception:
            model.generation_config = GenerationConfig()
            print("DEBUG: Using default GenerationConfig")
            
        print("DEBUG: Success!")
    except Exception as e:
        print("\n=== TRACEBACK START ===")
        traceback.print_exc()
        print("=== TRACEBACK END ===\n")

if __name__ == "__main__":
    debug_load()
