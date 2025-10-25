# diagnostic_script.py
import torch
from model.pro_model import Mymodel
from argparse import Namespace

MODEL_PATH = "/root/pycharm_semanticsearch/PATH_TO_OUTPUT_MODEL/webq/final_student_model_fp32"
args = Namespace(num_heads=8, ln=True, norm=True, padding_side='right', neg_K=3)

# Create model and get its parameter names
model = Mymodel(model_name_or_path=MODEL_PATH, args=args)
model_params = set(model.state_dict().keys())
print("Current model parameters (first 10):")
for i, param in enumerate(list(model_params)[:10]):
    print(f"  {i}: {param}")

# Load saved state dict and get its parameter names
saved_state = torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location="cpu")
saved_params = set(saved_state.keys())
print("\nSaved model parameters (first 10):")
for i, param in enumerate(list(saved_params)[:10]):
    print(f"  {i}: {param}")

# Find common parameters
common = model_params & saved_params
print(f"\nCommon parameters: {len(common)}")
for param in sorted(common):
    print(f"  ✓ {param}")