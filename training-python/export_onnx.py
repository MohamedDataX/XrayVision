import torch
import sys
sys.path.insert(0, "src")

from models.ssd_cnn_256 import create_model, NUM_CLASSES, IMAGE_SIZE

model = create_model(num_classes=NUM_CLASSES)

checkpoint = torch.load("models_fine_tuned/best_model.pt", map_location="cpu", weights_only=False)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict)
model.eval()

dummy = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE)

torch.onnx.export(
    model,
    dummy,
    "models_fine_tuned/best_model.onnx",
    input_names=["input"],
    output_names=["cls_preds", "reg_preds"],
    dynamic_axes={"input": {0: "batch"}},
    opset_version=17
)

print("Export OK → models_fine_tuned/best_model.onnx")