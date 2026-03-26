import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                out = model(x)
                probs = torch.softmax(out, dim=1)
                outputs.append(probs)
        avg_probs = torch.stack(outputs, dim=0).mean(dim=0)
        return avg_probs


def build_ensemble(model_classes, input_dim, num_classes, checkpoint_paths, device):
    models = []
    for name, path in checkpoint_paths.items():
        if name not in model_classes:
            print(f"Skipping unknown model: {name}")
            continue
        model = model_classes[name](input_dim, num_classes).to(device)
        try:
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
            models.append(model)
            print(f"Loaded {name} from {path}")
        except Exception as e:
            print(f"Failed to load {name}: {e}")

    if not models:
        raise ValueError("No models were loaded for ensemble")

    return EnsembleModel(models).to(device)
