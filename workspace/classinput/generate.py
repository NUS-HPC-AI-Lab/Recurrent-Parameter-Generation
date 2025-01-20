import sys, os, json
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Recurrent-Parameter-Generation") + 1])
sys.path.append(root)
os.chdir(root)


# torch
import time
import torch
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


def generate(save_path, embedding, need_test=True):
    print("\n==> Generating..")
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        prediction = None
        def display_time():
            while prediction is None:
                elapsed_time = time.time() - start_time
                elapsed_minutes = elapsed_time / 60
                print(f"Elapsed time: {elapsed_minutes:.2f} minutes", end="\r")
                time.sleep(0.1)
        import threading
        timer_thread = threading.Thread(target=display_time)
        timer_thread.start()
        prediction = model(sample=True, condition=embedding[None], permutation_state=False)
        timer_thread.join()
        print()
    generated_norm = torch.nanmean((prediction.cpu()).abs())
    print("Generated_norm:", generated_norm.item())
    if need_test:
        real_emb = input("Input your expected class (ONLY FOR EVALUATING): ")
        # real_emb = "[0,0,1,1,1,1,1,1,0,0]"
        real_emb = torch.tensor(eval(real_emb), dtype=torch.float)
        class_index = str(int("".join([str(int(i)) for i in real_emb]), 2)).zfill(4)
        train_set.save_params(prediction, save_path=save_path.format(class_index))
        print("Saved to:", save_path.format(class_index))
        test_command = os.path.join(test_set.test_command + save_path.format(class_index))
        os.system(test_command)
    model.train()
    return prediction
