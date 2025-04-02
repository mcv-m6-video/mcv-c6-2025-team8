import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from model.model_classification import Model
import json

# Load the configuration from baseline.json
config_path = "config/baseline.json"
with open(config_path, "r") as f:
    config = json.load(f)

# Initialize the model
model = Model(args=config)
model._model.eval()  # Set to evaluation mode

# Dummy input: (batch_size, clip_len, channels, height, width)
dummy_input = torch.randn(1, config["clip_len"], 3, 224, 398).to(config["device"])

# Compute FLOPs
flop_analyzer = FlopCountAnalysis(model._model, dummy_input)
gflops = flop_analyzer.total() / 1e9  # Convert to GFLOPs

# Compute trainable and total parameters
print(parameter_count_table(model._model))

trainable_params = sum(p.numel() for p in model._model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model._model.parameters())

# Print results
print(f"GFLOPs: {gflops:.2f}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Total Parameters: {total_params:,}")
