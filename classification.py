from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from transformers import AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt

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
    
    for_mask = np.full((head_len,head_len),int(head_len/2)*2+head_len)
    j2i_Ks = []
    j2i_Vs = []
    for l in range(layer_len):
        j2i_K = {str(k):k for k in range(head_len)}
        j2i_V = {str(k):k for k in range(head_len)}
        index_K = torch.topk(torch.reshape(collect["K"][l],(-1,)), int(head_len/2)*2+head_len, largest=False)[1].cpu()
        for i in index_K:
            row = int(i) // head_len
            clo = int(i) % head_len
            if row < clo:
                j2i_K[str(clo)] = j2i_K[str(row)]
        index_V = torch.topk(torch.reshape(collect["V"][l],(-1,)), int(head_len/2)*2+head_len, largest=False)[1].cpu()
        for i in index_V:
            row = int(i) // head_len
            clo = int(i) % head_len
            if row < clo:
                j2i_V[str(clo)] = j2i_V[str(row)]
        j2i_Ks.append(j2i_K)
        j2i_Vs.append(j2i_V)
    return j2i_Ks, j2i_Vs

def savefig(collect,l,name,div=1):
    head_len = collect["K"].shape[1]
    K = np.empty([head_len,head_len])
    V = np.empty([head_len,head_len])
    for i in range(head_len):
        for j in range(head_len):
            K[i][j] = collect["K"][l][i][j].cpu().numpy()/div
            V[i][j] = collect["V"][l][i][j].cpu().numpy()/div
    fig = plt.figure()
    ak = fig.add_subplot(121)
    cak = ak.matshow(K)
    fig.colorbar(cak)
    av = fig.add_subplot(122)
    cav = av.matshow(V)
    fig.colorbar(cav)
    # plt.savefig("/share/jpg/"+str(l)+".jpg")
    plt.savefig("/share/jpg/"+str(name)+".jpg")

# profile
generated = {}
generated["collect"] = {"K":torch.zeros([model.config.n_layers,model.config.n_heads,model.config.n_heads],dtype=torch.float64,device=device),"V":torch.zeros([model.config.n_layers,model.config.n_heads,model.config.n_heads],dtype=torch.float64,device=device)}
for i in range(10):
    input_ids = tokenized_imdb["train"][i]
    _, collect = forward(generated, input_ids)
    generated["collect"] = collect
    if i == 0:
        savefig(collect,0,i)
savefig(collect,0,"avg",10)
j2i_Ks, j2i_Vs = cut_half(collect)


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