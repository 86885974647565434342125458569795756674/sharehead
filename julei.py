from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from transformers import AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
import random

random.seed(100)

imdb = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True)

# print(len(tokenized_imdb["train"]))
# 25000
# 'input_ids', 'attention_mask'

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

device = "cuda:1"
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
).to(device)

def forward(generated, input_ids):
    generated["input_ids"] = torch.unsqueeze(torch.tensor(input_ids["input_ids"],device=device),0)
    generated["attention_mask"] = torch.unsqueeze(torch.tensor(input_ids["attention_mask"],device=device),0)
    with torch.no_grad():
        output, collect = model(**generated)
    return output, collect

def cut_half(collect):
    layer_len = collect["K"].shape[0]
    head_len = collect["K"].shape[1]
    collect["K"] = collect["K"].cpu()
    collect["V"] = collect["V"].cpu()
    
    old_zhixinK = []
    for l in range(layer_len):
        zhixin_index = random.sample([i for i in range(head_len)], int(head_len/2))
        old_zhixinK.append([collect["K"][l][i] for i in zhixin_index])
        while True:
            new_zhixin = [torch.zeros_like(collect["K"][l][0]) for _ in zhixin_index]
            new_zhixin_div = [0 for _ in zhixin_index]
            for i in range(head_len):
                min_juli = 99999999999
                i_zhixin_index = 0
                for j in len(zhixin_index):
                    juli = torch.norm(collect["K"][l][i]-old_zhixinK[l][j],p=2,dim=-1)
                    if juli < min_juli:
                        i_zhixin_index = j
                        min_juli = juli
                new_zhixin[i_zhixin_index] += collect["K"][l][i]
                new_zhixin_div[i_zhixin_index] += 1
            ok = True
            for i in range(len(zhixin_index)):
                if not new_zhixin[i]/new_zhixin_div[i].equal(old_zhixinK[l][i]):
                    ok = False
                    break
            old_zhixinK[l] = [new_zhixin[i]/new_zhixin_div[i] for i in range(len(zhixin_index))]
            if ok:
                break
    old_zhixinV = []
    for l in range(layer_len):
        zhixin_index = random.sample([i for i in range(head_len)], int(head_len/2))
        old_zhixinV.append([collect["V"][l][i] for i in zhixin_index])
        while True:
            new_zhixin = [torch.zeros_like(collect["V"][l][0]) for _ in zhixin_index]
            new_zhixin_div = [0 for _ in zhixin_index]
            for i in range(head_len):
                min_juli = 99999999999
                i_zhixin_index = 0
                for j in len(zhixin_index):
                    juli = torch.norm(collect["V"][l][i]-old_zhixinV[l][j],p=2,dim=-1)
                    if juli < min_juli:
                        i_zhixin_index = j
                        min_juli = juli
                new_zhixin[i_zhixin_index] += collect["V"][l][i]
                new_zhixin_div[i_zhixin_index] += 1
            ok = True
            for i in range(len(zhixin_index)):
                if not new_zhixin[i]/new_zhixin_div[i].equal(old_zhixinV[l][i]):
                    ok = False
                    break
            old_zhixinV[l] = [new_zhixin[i]/new_zhixin_div[i] for i in range(len(zhixin_index))]
            if ok:
                break
    return old_zhixinK, old_zhixinV
    
# profile
generated = {}
generated["collect"] = {"K":torch.zeros([model.config.n_layers,model.config.n_heads,model.config.dim//model.config.n_heads],dtype=torch.float64,device=device),"V":torch.zeros([model.config.n_layers,model.config.n_heads,model.config.dim//model.config.n_heads],dtype=torch.float64,device=device)}
for i in range(10):
    input_ids = tokenized_imdb["train"][i]
    _, collect = forward(generated, input_ids)
    generated["collect"] = collect

newK, newV = cut_half(collect)


# test
# generated = {"j2i_K":j2i_Ks,"j2i_V":j2i_Vs}
# for i in range(10,1000):
#     input_ids = tokenized_imdb["train"][i]
#     output, _ = forward(generated, input_ids)
#     _, _ = output.cpu()
   
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)


# inputs = tokenizer(text, return_tensors="pt")

# with torch.no_grad():
#     logits = model(**inputs).logits

# predicted_class_id = logits.argmax().item()
# model.config.id2label[predicted_class_id]