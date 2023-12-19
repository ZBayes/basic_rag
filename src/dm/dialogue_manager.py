# coding=utf-8
# Filename:    dialogue_manager.py
# Author:      ZENGGUANRONG
# Date:        2023-12-19
# description: 对话管理，核心对话流程管理模块

import requests
from loguru import logger
from src.server.client import run_client

RAG_PROMPT = """请根据用户提问和参考资料进行回复，回复的内容控制在100字左右。

用户提问：{}

参考材料：
{}"""

class DialogueManager:
    def __init__(self, config):
        self.config = config["config"]
        logger.info(self.config)

    def predict(self, query):
        # 1. 预处理
        
        # 2. 请求检索
        retrieval_result = run_client(self.config["search_url"], query)
        logger.info("[DialogueManager] retrieval_result: {}".format(retrieval_result))
        
        retrieval_answer = ""
        if len(retrieval_result.get("answer", [])) > 0:
            # 此处只取第一名
            retrieval_answer = retrieval_result["answer"][0]["answer"]
        logger.info("[DialogueManager] retrieval_answer: {}".format(retrieval_answer))
        
        # 3. 请求大模型
        prompt = self.build_llm_prompt(query, retrieval_answer)
        logger.info("[DialogueManager] prompt: {}".format(prompt))
        llm_result = run_client(self.config["llm_url"], prompt)
        logger.info("[DialogueManager] llm_result: {}".format(llm_result))
        return llm_result.get("answer", "")
        
    def build_llm_prompt(self, query, retrieval_answer):
        
        return RAG_PROMPT.format(query, retrieval_answer)