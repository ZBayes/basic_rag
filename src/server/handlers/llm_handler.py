# coding=utf-8
# Filename:    llm_handler.py
# Author:      ZENGGUANRONG
# Date:        2023-12-17
# description: 大模型服务handler

import json,copy
from loguru import logger

from tornado.escape import json_decode
import tornado.ioloop
from tornado.web import RequestHandler, Application
from tornado.httpserver import HTTPServer
from tornado.options import options, define

from src.models.llm.llm_model import LlmModel

class LlmHandler(RequestHandler):
    
    def initialize(self, llm_model:LlmModel):
        self.llm_model = llm_model
        # request_body:
        # {
        #   "query": "什么人不能吃花生"
        # }

        # response_body = {
        #     "answer":"XXX"
        # }

    async def post(self):
        answer = self.llm_model.predict(json_decode(self.request.body).get("query", ""))
        response_body = {"answer": answer[0]}
        self.write(response_body)

def StartLlmHandler(request_config: dict, llm_model: LlmModel):
    # 启动TestClass服务的进程
    # 注：如果是同端口，只是不同的url路径，则直接都放在handler_routes里面即可
    # 注：test_class需要在外面初始化后再传进来，而不能在initialize里面加载，initialize是每次请求都会执行一遍，例如Searcher下的模型类，肯定不能在这里面修改
    handler_routes = [(request_config["url_suffix"], LlmHandler, {"llm_model":llm_model})]
    app = Application(handlers=handler_routes)
    http_server = HTTPServer(app)
    http_server.listen(request_config["port"])
    tornado.ioloop.IOLoop.current().start()