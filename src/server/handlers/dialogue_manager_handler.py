# coding=utf-8
# Filename:    dialogue_manager_handler.pyt
# Author:      ZENGGUANRONG
# Date:        2023-12-19
# description: 对话管理handler

import json,copy
from loguru import logger

from tornado.escape import json_decode
import tornado.ioloop
from tornado.web import RequestHandler, Application
from tornado.httpserver import HTTPServer
from tornado.options import options, define

from src.dm.dialogue_manager import DialogueManager

class DialogueManagerHandler(RequestHandler):
    def initialize(self, dialogue_manager:DialogueManager):
        self.dialogue_manager = dialogue_manager
        # request_body:
        # {
        #   "query": "什么人不能吃花生"
        # }

        # response_body = {
        #     "answer":"XXX"
        # }

    async def post(self):
        answer = self.dialogue_manager.predict(json_decode(self.request.body).get("query", ""))
        logger.info(answer)
        response_body = {"answer": answer}
        self.write(response_body)

def StartDialogueManagerHandler(request_config: dict, dialogue_manager: DialogueManager):
    # 启动TestClass服务的进程
    # 注：如果是同端口，只是不同的url路径，则直接都放在handler_routes里面即可
    # 注：test_class需要在外面初始化后再传进来，而不能在initialize里面加载，initialize是每次请求都会执行一遍，例如Searcher下的模型类，肯定不能在这里面修改
    handler_routes = [(request_config["url_suffix"], DialogueManagerHandler, {"dialogue_manager":dialogue_manager})]
    app = Application(handlers=handler_routes)
    http_server = HTTPServer(app)
    http_server.listen(request_config["port"])
    tornado.ioloop.IOLoop.current().start()