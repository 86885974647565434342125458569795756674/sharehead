from transformers import AutoModel
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy

device = "cuda:1"
dataset = load_dataset("snoop2head/commoncrawl_sampled_gpt2-xl")
model = AutoModel.from_pretrained("gpt2").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")


generated = tokenizer(dataset["train"][100]["generated"], return_tensors="pt").to(device)
generated["collect"] = {"K":{},"V":{}}
with torch.no_grad():
    output, collect = model(**generated)

layer_len = len(collect["K"])
head_len = len(collect["K"]["0"])
K = numpy.empty([head_len,head_len])
V = numpy.empty([head_len,head_len])
for l in range(layer_len):
    for i in range(head_len):
        print("K",str(l),str(i),collect["K"][str(l)][str(i)]["absolute"].cpu(),flush=True)
        print("V",str(l),str(i),collect["V"][str(l)][str(i)]["absolute"].cpu(),flush=True)
        for j in range(head_len):
            K[i][j] = collect["K"][str(l)][str(i)][str(j)].cpu().numpy()
            V[i][j] = collect["V"][str(l)][str(i)][str(j)].cpu().numpy()
    fig = plt.figure()
    ak = fig.add_subplot(121)
    cak = ak.matshow(K)
    fig.colorbar(cak)
    av = fig.add_subplot(122)
    cav = av.matshow(V)
    fig.colorbar(cav)
    plt.savefig("/share/jpg100/"+str(l)+".jpg")


