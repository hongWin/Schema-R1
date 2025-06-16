# Schema-R1

<img src="assets/logo.png" width="400">

## Training Plots
<img src="assets/output.png" width="800">

## Setup
```shell
conda create -n schema-R1 python=3.11 && conda create activate schema-R1 && pip install --upgrade pip
```

Next, install vLLM and FlashAttention:

```shell
pip install vllm==0.8.4
pip install setuptools && pip install flash-attn --no-build-isolation
Recommend manual download and installation [flash-attn](https://github.com/Dao-AILab/flash-attention/releases)
pip install swanlab==0.5.7 
```
Tips, check the transformer version while fail to start VLLM:
```shell
pip install transformers==4.51.3
pip install trl==0.18.0
```


## Evaluation
ALL eval process can be used in Schema-R1/src/eval/

## Acknowledegments

[Smol-r1](https://github.com/rasdani/smolR1)

[Open-r1](https://github.com/huggingface/open-r1)

[DTS-SQL](https://github.com/MohammadrezaPourreza/DTS-SQL)
