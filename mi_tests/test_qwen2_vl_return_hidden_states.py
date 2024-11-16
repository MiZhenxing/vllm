from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from vllm.attention import AttentionMetadata
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.qwen2 import Qwen2Model, Qwen2ForCausalLM
# from vllm.model_executor.models.qwen2_vl import Qwen2VLForConditionalGeneration
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import PoolerOutput
from vllm.model_executor.models.utils import is_pp_missing_parameter, make_layers

from PIL import Image
import requests
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


from vllm import LLM, SamplingParams
import types

# Class to store embeddings and manage hook
class EmbeddingExtractor:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.embeddings = []
        self.register_hook()

    # Hook function to capture embeddings from the final layer
    def hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        self.embeddings.append(output)

    # Register the hook
    def register_hook(self):
        layer = dict([*self.model.named_modules()])[self.layer_name]
        layer.register_forward_hook(self.hook_fn)

    # Retrieve embeddings after forward pass
    def get_embeddings(self):
        return self.embeddings

    def reset_embeddings(self):
        self.embeddings = []


# Create an LLM.
# model = LLM(model="Qwen/Qwen2-VL-7B-Instruct", enforce_eager=True, dtype="bfloat16", return_hidden_states=True)


# inner_model = model.llm_engine.model_executor.driver_worker.model_runner.model

question = ["describe the image", "What can you see in this image?"]
prompt = ["<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
              f"{question[i]}<|im_end|>\n"
              "<|im_start|>assistant\n" for i in range(2)]


# url = "https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/refs/heads/main/assets/images/vermeer.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

url = "/group_nfs/usr/zmi/personalization/MiniGPT-4/imgs/vermeer.jpg"
image = Image.open(url).convert("RGB")

messages = [[
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                "image": image,
            },
            {"type": "text", "text": "describe the image"},
        ],
    },
    {
        "role": "assistant",
        "content": "This image shows"
    }
],
[
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                "image": image,
            },
            {"type": "text", "text": "What can you see in this image?"},
        ],
    },
    {
        "role": "assistant",
        # "content": "I see in this image"
        "content": "This image shows"
    }
],
]

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Preparation for inference
text = processor.apply_chat_template(
    # messages, tokenize=False, add_generation_prompt=True, continue_final_message=False
    messages, tokenize=False, add_generation_prompt=False, continue_final_message=True
)

inputs = [{
        "prompt": text[0],
        "multi_modal_data": {
            "image": image
        },
    },
    {
        "prompt": text[0],
        "multi_modal_data": {
            "image": image
        },
    },
    ]

sampling_params = SamplingParams(temperature=0.6,
    top_p=0.9,
    max_tokens=256,
    min_tokens=256,
    ignore_eos=True,
    stop_token_ids=None)


model = LLM(model="Qwen/Qwen2-VL-7B-Instruct", enforce_eager=True, dtype="bfloat16", gpu_memory_utilization=0.5, return_hidden_states=True)
inner_model = model.llm_engine.model_executor.driver_worker.model_runner.model

# Register the hook on the final layer using the EmbeddingExtractor class
extractor = EmbeddingExtractor(inner_model, "model.norm")  # Adjust layer name

extractor.reset_embeddings()

outputs_gene = model.generate(inputs, sampling_params=sampling_params)
# Get the embeddings from the final layer
embeddings = extractor.get_embeddings()

pass

batch_size = 2



output_embed_1 = torch.cat([i.resize(batch_size, i.shape[0] // batch_size, i.shape[-1]) for i in embeddings[1:]], dim=1).cpu()
output_embed_2 = torch.stack([outputs_gene[i].outputs[0].hidden_states[1:] for i in range(batch_size)])
assert torch.all(output_embed_1 == output_embed_2)

# assert torch.all(outputs_gene[0].prompt_hidden_states[-1] == outputs_gene[0].outputs[0].hidden_states[0])