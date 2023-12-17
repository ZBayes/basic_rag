# coding=utf-8
# Filename:    search_handler.py
# Author:      ZENGGUANRONG
# Date:        2023-12-12
# description: 检索器服务

import json,copy
from loguru import logger

from tornado.escape import json_decode
import tornado.ioloop
from tornado.web import RequestHandler, Application
from tornado.httpserver import HTTPServer
from tornado.options import options, define

from src.searcher.searcher import Searcher

class SearcherHandler(RequestHandler):
    
    def initialize(self, searcher:Searcher):
        self.searcher = searcher
        # request_body:
        # {
        #   "query": "什么人不能吃花生"
        # }

        # response_body = {
        #     "answer":[{
        #         "match_query":"match_query",
        #         "answer":"answer",
        #         "score":"score"
        #     }]
        # }

    async def post(self):
        answers = self.searcher.search(json_decode(self.request.body).get("query", ""))
        result = []
        for answer in answers:
            tmp_result = {}
            # tmp_result["query"] = answer[0]
            tmp_result["answer"] = answer[1][1]["answer"]
            tmp_result["match_query"] = answer[1][0]
            tmp_result["score"] = str(answer[3])
            result.append(copy.deepcopy(tmp_result))
        response_body = {"answer": result}
        self.write(response_body)

def StartSearcherHandler(request_config: dict, searcher: Searcher):
    # 启动TestClass服务的进程
    # 注：如果是同端口，只是不同的url路径，则直接都放在handler_routes里面即可
    # 注：test_class需要在外面初始化后再传进来，而不能在initialize里面加载，initialize是每次请求都会执行一遍，例如Searcher下的模型类，肯定不能在这里面修改
    handler_routes = [(request_config["url_suffix"], SearcherHandler, {"searcher":searcher})]
    app = Application(handlers=handler_routes)
    http_server = HTTPServer(app)
    http_server.listen(request_config["port"])
    tornado.ioloop.IOLoop.current().start()