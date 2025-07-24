import torch
import numpy as np
import os
from src.padain_synthesis.trainer import PadainSynthesisModel, PadainSynthesisConfig, load_trained_model

class PadainSynthesisPipeline:
    def __init__(self, model, config, device="cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model.eval()

    @classmethod
    def from_pretrained(cls, checkpoint_path, config_path=None, device="cuda"):
        # config 자동 로드
        if config_path is None:
            config_path = os.path.join(checkpoint_path, "config.json")
        config = PadainSynthesisConfig.from_json_file(config_path)
        model = load_trained_model(checkpoint_path, config=config)
        return cls(model, config, device=device)

    def __call__(self, input, reference):
        # input, reference: numpy array (1, H, W)
        import torch
        merged_input = torch.cat([torch.from_numpy(input), torch.from_numpy(reference)], dim=0).unsqueeze(0).to(self.device)
        print('[pipeline.py] merged_input shape:', merged_input.shape)
        with torch.no_grad():
            output = self.model(merged_input).last_hidden_state[0].cpu().numpy()
        print('[pipeline.py] output shape:', output.shape)
        return output 