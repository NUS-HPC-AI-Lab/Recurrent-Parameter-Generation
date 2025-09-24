import sys, os, json
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Recurrent-Parameter-Generation")+1])
sys.path.append(root)
os.chdir(root)
with open("./workspace/config.json", "r") as f:
    additional_config = json.load(f)
USE_WANDB = additional_config["use_wandb"]

# set global seed
import random
import numpy as np
import torch
seed = SEED = 999
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

# other
import math
import random
import warnings
from _thread import start_new_thread
warnings.filterwarnings("ignore", category=UserWarning)
if USE_WANDB: import wandb
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.cuda.amp import autocast
# model
from model.pdiff import PDiff as Model
from model.pdiff import OneDimVAE as VAE
from model.diffusion import DDPMSampler, DDIMSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import AutocastKwargs
from accelerate import Accelerator
# dataset
from dataset import Cifar10_CNNMedium as Dataset
from torch.utils.data import DataLoader




config = {
    "seed": SEED,
    # dataset setting
    "dataset": Dataset,
    "sequence_length": 'auto',
    # train setting
    "batch_size": 16,
    "num_workers": 16,
    "total_steps": 40000,
    "vae_steps": 10000,
    "learning_rate": 0.0001,
    "vae_learning_rate": 0.00001,
    "weight_decay": 0.0,
    "save_every": 40000//5,
    "print_every": 50,
    "autocast": lambda i: 5000 < i < 45000,
    "checkpoint_save_path": "./checkpoint",
    # test setting
    "test_batch_size": 1,  # fixed, don't change this
    "generated_path": Dataset.generated_path,
    "test_command": Dataset.test_command,
    # to log
    "model_config": {
        # diffusion config
        "layer_channels": [1, 64, 128, 256, 512, 256, 128, 64, 1],
        "model_dim": 1024,
        "kernel_size": 7,
        "sample_mode": DDPMSampler,
        "beta": (0.0001, 0.02),
        "T": 1000,
        # vae config
        "channels": [64, 256, 384, 512, 64],
    },
    "tag": "compare_pdiff_cnnsmedium_vae",
}




# Data
divide_slice_length = int(2 ** len(config["model_config"]["channels"]))
print('==> Preparing data..')
train_set = config["dataset"](dim_per_token=divide_slice_length,
                              granularity=0,
                              pe_granularity=0,
                              fill_value=0.)
print("Dataset length:", train_set.real_length)
print("input shape:", train_set[0][0].flatten().shape)
if config["sequence_length"] == "auto":
    config["sequence_length"] = train_set.sequence_length * divide_slice_length
    print(f"sequence length: {config['sequence_length']}")
train_loader = DataLoader(
    dataset=train_set,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    persistent_workers=True,
    drop_last=True,
    shuffle=True,
)

# Model
print('==> Building model..')
Model.config = config["model_config"]
model = Model(sequence_length=config["sequence_length"])  # model setting is in model
vae = VAE(d_model=config["model_config"]["channels"],
          d_latent=config["model_config"]["model_dim"],
          sequence_length=config["sequence_length"],
          kernel_size=config["model_config"]["kernel_size"])

# Optimizer
print('==> Building optimizer..')
vae_optimizer = optim.AdamW(
    params=vae.parameters(),
    lr=config["vae_learning_rate"],
    weight_decay=config["weight_decay"],
)
optimizer = optim.AdamW(
    params=model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)
vae_scheduler = CosineAnnealingLR(
    optimizer=vae_optimizer,
    T_max=config["vae_steps"],
)
scheduler = CosineAnnealingLR(
    optimizer=optimizer,
    T_max=config["total_steps"],
)

# accelerator
if __name__ == "__main__":
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs,])
    vae, model, vae_optimizer, optimizer, train_loader = \
            accelerator.prepare(vae, model, vae_optimizer, optimizer, train_loader)


# wandb
if __name__ == "__main__" and USE_WANDB and accelerator.is_main_process:
    wandb.login(key=additional_config["wandb_api_key"])
    wandb.init(project="Recurrent-Parameter-Generation", name=config['tag'], config=config,)




# Training
print('==> Defining training..')

def train_vae():
    if not USE_WANDB:
        train_loss = 0
        this_steps = 0
    print("==> Start training vae..")
    vae.train()
    for batch_idx, (param, _) in enumerate(train_loader):
        vae_optimizer.zero_grad()
        # train
        # noinspection PyArgumentList
        with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=config["autocast"](batch_idx))):
            param = param.flatten(start_dim=1)
            loss = vae(x=param)
        accelerator.backward(loss)
        vae_optimizer.step()
        vae_scheduler.step(batch_idx)
        # to logging losses and print and save
        if USE_WANDB and accelerator.is_main_process:
            wandb.log({"vae_loss": loss.item()})
        elif USE_WANDB:
            pass  # don't print
        else:  # not use wandb
            train_loss += loss.item()
            this_steps += 1
            if this_steps % config["print_every"] == 0:
                print('Loss: %.6f' % (train_loss/this_steps))
                this_steps = 0
                train_loss = 0
        if batch_idx >= config["vae_steps"]:
            break

def train():
    if not USE_WANDB:
        train_loss = 0
        this_steps = 0
    print("==> start training..")
    model.train()
    for batch_idx, (param, _) in enumerate(train_loader):
        optimizer.zero_grad()
        # train
        # noinspection PyArgumentList
        with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=config["autocast"](batch_idx))):
            param = param.flatten(start_dim=1)
            with torch.no_grad():
                mu, _ = vae.encode(param)
            loss = model(x=mu)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step(batch_idx)
        # to logging losses and print and save
        if USE_WANDB and accelerator.is_main_process:
            wandb.log({"train_loss": loss.item()})
        elif USE_WANDB:
            pass  # don't print
        else:  # not use wandb
            train_loss += loss.item()
            this_steps += 1
            if this_steps % config["print_every"] == 0:
                print('Loss: %.6f' % (train_loss/this_steps))
                this_steps = 0
                train_loss = 0
        if batch_idx % config["save_every"] == 0 and accelerator.is_main_process:
            os.makedirs(config["checkpoint_save_path"], exist_ok=True)
            state = {"diffusion": accelerator.unwrap_model(model).state_dict(), "vae": vae.state_dict()}
            torch.save(state, os.path.join(config["checkpoint_save_path"], config["tag"]+".pth"))
            generate(save_path=config["generated_path"], need_test=True)
        if batch_idx >= config["total_steps"]:
            break

def generate(save_path=config["generated_path"], need_test=True):
    print("\n==> Generating..")
    model.eval()
    with torch.no_grad():
        mu = model(sample=True)
        prediction = vae.decode(mu)
        generated_norm = prediction.abs().mean()
    print("Generated_norm:", generated_norm.item())
    if USE_WANDB:
        wandb.log({"generated_norm": generated_norm.item()})
    prediction = prediction.view(-1, divide_slice_length)
    train_set.save_params(prediction, save_path=save_path)
    if need_test:
        start_new_thread(os.system, (config["test_command"],))
    model.train()
    return prediction




if __name__ == '__main__':
    train_vae()
    vae = accelerator.unwrap_model(vae)
    train()
    del train_loader  # deal problems by dataloader
    print("Finished Training!")
    exit(0)