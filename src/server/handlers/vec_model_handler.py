# coding=utf-8
# Filename:    vec_model_handler.py
# Author:      ZENGGUANRONG
# Date:        2023-12-13
# description: 模型服务

import json,copy
from loguru import logger

from tornado.escape import json_decode
import tornado.ioloop
from tornado.web import RequestHandler, Application
from tornado.httpserver import HTTPServer
from tornado.options import options, define

from src.models.vec_model.vec_model import VectorizeModel

class VecModelHandler(RequestHandler):
    
    def initialize(self, vec_model:VectorizeModel):
        self.vec_model = vec_model
        # request_body:
        # {
        #   "query": "什么人不能吃花生"
        # }

        # response_body = {
        #     "answer":[{
        #         "query":"query",
        #         "answer":"answer",
        #         "score":"score"
        #     }]
        # }

    async def post(self):
        vec_result = self.vec_model.predict_vec_request(json_decode(self.request.body).get("query", ""))
        response_body = {"vec_result": vec_result}
        self.write(response_body)

def StartVecModelHandler(request_config: dict, vec_model: VectorizeModel):
    # 启动TestClass服务的进程
    # 注：如果是同端口，只是不同的url路径，则直接都放在handler_routes里面即可
    # 注：test_class需要在外面初始化后再传进来，而不能在initialize里面加载，initialize是每次请求都会执行一遍，例如知识问答的索引更新，肯定不能在这里面修改
    handler_routes = [(request_config["url_suffix"], VecModelHandler, {"vec_model":vec_model})]
    app = Application(handlers=handler_routes)
    http_server = HTTPServer(app)
    http_server.listen(request_config["port"])
    tornado.ioloop.IOLoop.current().start()