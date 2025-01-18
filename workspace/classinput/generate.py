import sys, os, json
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Recurrent-Parameter-Generation")+1])
sys.path.append(root)
os.chdir(root)

# torch
import torch
import random
from torch import nn
# father
from workspace.classinput import generalization as item
train_set = item.train_set
test_set = item.test_set
train_set.set_infinite_dataset(max_num=train_set.real_length)
print("num_generated:", test_set.real_length)
config = item.config
model = item.model
model.config["diffusion_batch"] = 128
assert config.get("tag") is not None, "Remember to set a tag."


# Model
print('==> Building model..')
diction = torch.load("./checkpoint/generalization.pth")
permutation_shape = diction["to_permutation_state.weight"].shape
model.to_permutation_state = nn.Embedding(*permutation_shape)
model.load_state_dict(diction)
model = model.cuda()


# generate
print('==> Defining generate..')
def generate(save_path, embedding, real_embedding, need_test=True):
    class_index = str(int("".join([str(int(i)) for i in real_embedding]), 2)).zfill(4)
    print("\n==> Generating..")
    model.eval()
    with torch.no_grad():
        prediction = model(sample=True, condition=embedding[None], permutation_state=False)
        generated_norm = torch.nanmean((prediction.cpu()).abs())
    print("Generated_norm:", generated_norm.item())
    train_set.save_params(prediction, save_path=save_path.format(class_index))
    print("Saved to:", save_path.format(class_index))
    if need_test:
        test_command = os.path.join(test_set.test_command + save_path.format(class_index))
        os.system(test_command)
    model.train()
    return prediction




# if __name__ == "__main__":
#     for i in range(config["num_generated"]):
#         if config["specific_item"] is not None:
#             assert isinstance(config["specific_item"], int)
#             i = config["specific_item"]
#             print(f"generate index {i}\n")
#         print("Save to", config["generated_path"].format(config["tag"], "class####"))
#         generate(
#             save_path=config["generated_path"],
#             test_command=config["test_command"],
#             need_test=config["need_test"],
#             index=random.randint(0, len(train_set)-1) if config["specific_item"] is None else i,
#         )
