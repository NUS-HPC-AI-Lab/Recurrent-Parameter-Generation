import torch.nn as nn
import timm


def Model():
    model = timm.create_model("convnext_atto", pretrained=True)
    return model, model.head.fc


if __name__ == "__main__":
    model, _ = Model()
    print(model)
    num_param = 0
    for v in model.parameters():
        num_param += v.numel()
    print("num_param:", num_param)
