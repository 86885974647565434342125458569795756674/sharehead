docker run --privileged -it -p 15003:22 -p 15004:6006 --name cyy_sh --gpus device=all -v /data/cyy/transformers/sharehead:/share nvcr.io/nvidia/pytorch:23.09-py3

docker exec --privileged -it cyy_sh /bin/bash 

passwd

apt update

apt install openssh-server -y

vim /etc/ssh/sshd_config

PermitRootLogin yes

service ssh start

pip install transformers datasets

# gpt

计算不同head的KV，计算不同head对应token的相似度（距离的模），取平均

/usr/local/lib/python3.10/dist-packages/transformers/models/gpt2/modeling_gpt2.py

GPT2Model

GPT2Block

GPT2Attention

[1,145,768]

(batch, head, seq_length, head_features)

1, 12, 145, 64

/usr/local/lib/python3.10/dist-packages/transformers/pytorch_utils.py

[145,768] [768,2304] + [2304] = [145,2304] 

[1,145,2304]

[1,145,768]

qkv按列切分[768,2304/3]

[1,145,12,64]

不同head按列切分[768,2304/3/12]

/usr/bin/python /share/main.py > /share/absolute.txt

main

modeling_gpt

# bert

/usr/local/lib/python3.10/dist-packages/transformers/models/distilbert/modeling_distilbert.py

DistilBertForSequenceClassification.config

DistilBertModel

Transformer

TransformerBlock

MultiHeadSelfAttention

config.n_layers

config.n_heads

classification

modeling_distilbert

# 聚类

self.dim = config.dim

dim_per_head = self.dim // self.n_heads

kv的对应:

k:(1,2),(3,4)

v:(1,3),(2,4)

kv:(1,2)(1,3),(1,2)(2,4),(3,4)(1,3),(3,4)(2,4)

最后把head拼接在一起，再矩阵乘法，维度变小了，如何对应
