# coding=utf-8
# Filename:    llm_model.py
# Author:      ZENGGUANRONG
# Date:        2023-12-17
# description: 大模型调用模块，这里默认用的chatglm2

from transformers import AutoModel, AutoTokenizer
from typing import Tuple, List

class LlmModel:
    def __init__(self, model_path, config = {}):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().quantize(8).cuda()
        # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
        # from utils import load_model_on_gpus
        # model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
        self.model = self.model.eval()

        self.config = self._read_config_(config)
    
    def _read_config_(self, config):
        tmp_config = {}
        tmp_config["max_length"] = config.get("max_length", 2048)
        tmp_config["num_beams"] = config.get("num_beams", 1)
        tmp_config["do_sample"] = config.get("do_sample", True)
        tmp_config["top_p"] = config.get("top_p", 0.8)
        tmp_config["temperature"] = config.get("temperature", 0.8)
        return tmp_config

    def _chat(self, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, num_beams=1,
            do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None, **kwargs):
        chat_result = self.model.chat(self.tokenizer, query, history, max_length, num_beams, do_sample, top_p, temperature, logits_processor, **kwargs)
        return chat_result

    def predict(self, query):
        return self._chat(query, [], max_length=self.config["max_length"],
                                        num_beams=self.config["num_beams"],
                                        do_sample=self.config["do_sample"],
                                        top_p=self.config["top_p"],
                                        temperature=self.config["temperature"])

if __name__ == "__main__":
    path = "C:\\work\\tool\\chatglm2-6b"
    llm_model = LlmModel(path)
    print(llm_model.predict("今天吃什么，帮我推荐一下"))
