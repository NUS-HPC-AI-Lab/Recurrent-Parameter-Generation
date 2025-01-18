from transformers import AutoModelForCausalLM, AutoTokenizer
import os


model_name = os.path.join(os.path.dirname(__file__), "Qwen25llm")


print("Downloading Qwen2.5 files...")
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen2.5-3B-Instruct",
                  repo_type="model",
                  cache_dir=model_name,
                  local_dir_use_symlinks=False,
                  resume_download=True)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)  # load model
tokenizer = AutoTokenizer.from_pretrained(model_name)


def describe(prompt, system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]  # construct msgs
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )  # get text
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )  # generate
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]  # generate
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def discriminate(class_name, prompt):
    system_prompt = "You are an accurate discriminator. " \
                    "You need to determines if the class name matches the description. " \
                    "Answer with YES or NO."
    keywords = [word for word in prompt.split(" ")
                if "select" in word or "classif" in word or "find" in word or "all" in word]
    if len(keywords) == 0:
        description = prompt
    else:  # # len(keywords > 0)
        description = prompt.rsplit(keywords[-1], 1)[-1]
    prompt = f"Does the {class_name} belong to \"{description}\"? \n\nAnswer me with YES or NO."
    result = describe(prompt, system_prompt)
    if "NO" in result or "no" in result or "No" in result:
        return False
    else:  # assert YES in result
        return True


def get_embedding(prompt):
    class_names = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    results = []
    for class_name in class_names:
        result = discriminate(class_name, prompt)
        results.append(result)
    return results
