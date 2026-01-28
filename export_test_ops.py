import torch
import torch.nn as nn
import os

class InstanceNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.InstanceNorm1d(256, affine=True)
    def forward(self, x):
        return self.norm(x)

def export_test_models():
    os.makedirs("tests/rvc_ops", exist_ok=True)
    
    # 1. InstanceNorm
    model = InstanceNormModel()
    x = torch.randn(1, 256, 49)
    torch.onnx.export(model, x, "tests/rvc_ops/instance_norm.onnx", input_names=["input"], output_names=["output"], opset_version=14)
    
    # 2. RandomNormalLike
    class RandomModel(nn.Module):
        def forward(self, x):
            return torch.randn_like(x)
    torch.onnx.export(RandomModel(), x, "tests/rvc_ops/random_like.onnx", input_names=["input"], output_names=["output"], opset_version=14)

    print("Test models exported.")

if __name__ == "__main__":
    export_test_models()
