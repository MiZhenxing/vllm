### Introduction

This is a fork of vLLM for extracting embeddings of generated tokens efficiently. It is used in our [ThinkDiff](https://github.com/MiZhenxing/ThinkDiff) paper.

The modification is largely based on https://github.com/vllm-project/vllm/pull/7892/files. Many thanks to the author.
Since the original vLLM is under quick development, only `vllm==0.6.3.post1` is supported in this code. It would be great if someone could transfer https://github.com/vllm-project/vllm/pull/7892/files to the newest VLLM.

### Install


The first step is to install the original vLLM wheel:

```
pip install vllm==0.6.3.post1
```

Then you need to clone and install this code:

```
git clone https://github.com/MiZhenxing/vllm
cd vllm
```
Please make sure you are under the `return_hidden_states` branch.

Then only install the Python codes:

```
python python_only_dev.py
```
