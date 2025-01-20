import sys, os, json
root = os.sep + os.sep.join(__file__.split(os.sep)[1:__file__.split(os.sep).index("Recurrent-Parameter-Generation")+1])
sys.path.append(root)
os.chdir(root)


from workspace.classinput.generate import generate
from workspace.classinput.qwen25llm import get_embedding
import torch
import time




while True:
    time.sleep(0.5)
    save_name = "./workspace/classinput/generated_class{}.pth"
    print("\n\n\n==================================================================================")
    print('class includes: ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")')
    text_emb = input("Input your description: ") or "Give me a model to select all living things."
    # text_emb = "Give me a model to select all living things."
    emb = get_embedding(prompt=text_emb)
    emb = torch.tensor(emb, dtype=torch.float)
    params = generate(save_path=save_name, embedding=emb)
