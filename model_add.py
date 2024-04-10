from haystack.nodes import PromptModelInvocationLayer
from haystack.nodes.prompt.invocation_layer import DefaultTokenStreamingHandler
from llama_cpp import Llama
import os
from typing import Dict, List, Union, Type, Optional

import logging

logger = logging.getLogger(__name__)

class LlamaCPPLayer(PromptModelInvocationLayer):
    def __init__(self, model_location: Union[str, os.PathLike],
                 max_gen_length: Optional[int] = 128,
                 max_input_length: Optional[int] = 2048,
                 split_parts: Optional[int] = -1,
                 generation_seed: Optional[int] = 1337,
                 use_f16_kv: Optional[bool] = True,
                 return_all_logits: Optional[bool] = False,
                 restrict_vocab: Optional[bool] = False,
                 enable_mmap: Optional[bool] = True,
                 lock_memory: Optional[bool] = False,
                 enable_embedding: Optional[bool] = False,
                 thread_count: Optional[int] = None,
                 batch_size: Optional[int] = 512,
                 token_window_size: Optional[int] = 64,
                 lora_base_dir: Optional[str] = None,
                 lora_files_path: Optional[str] = None,
                 is_verbose: Optional[bool] = True,
                 **extra_args):

        if not model_location or not len(model_location):
            raise ValueError("Model location cannot be None or empty")

        self.model_location = model_location
        self.max_input_length = max_input_length
        self.max_gen_length = max_gen_length
        self.split_parts = split_parts
        self.generation_seed = generation_seed
        self.use_f16_kv = use_f16_kv
        self.return_all_logits = return_all_logits
        self.restrict_vocab = restrict_vocab
        self.enable_mmap = enable_mmap
        self.lock_memory = lock_memory
        self.enable_embedding = enable_embedding
        self.thread_count = thread_count
        self.batch_size = batch_size
        self.token_window_size = token_window_size
        self.lora_base_dir = lora_base_dir
        self.lora_files_path = lora_files_path
        self.is_verbose = is_verbose
        super().__init__(**extra_args)

    def _generate_prompt(self, query: str) -> str:
        prompt = super()._generate_prompt(query)
        return prompt

    def invoke(self, *args, **kwargs) -> List[str]:
        output_texts = []
        stream_mode = kwargs.pop("stream", False)

        generated_texts = []
        
        if kwargs and "prompt" in kwargs:
            prompt_text = kwargs.pop("prompt")
            model_args = {
                key: kwargs[key]
                for key in [
                    "suffix",
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "logprobs",
                    "echo",
                    "penalty",
                    "top_k",
                    "stop"
                ]
                if key in kwargs
            }
            
        if stream_mode:
            for token in self.model(prompt_text, stream=True, **model_args):
                generated_texts.append(token['choices'][0]['text'])
        else:
            model_output = self.model(prompt_text, **model_args)
            generated_texts = [text['text'] for text in model_output['choices']]
        return generated_texts

    @classmethod
    def supports(cls, model_identifier: str, **kwargs) -> bool:
        return model_identifier is not None and len(model_identifier) > 0
