# coding=utf-8
# Filename:    test_class_client.py
# Author:      ZENGGUANRONG
# Date:        2023-09-10
# description: 测试用的客户端

import numpy as np
import json,requests,time,random
from multiprocessing import Pool

def run_client(url, query):
    response = requests.post(url, json.dumps({"query": query}))
    return json.loads(response.text)

def cal_time_result(time_list):
    tp = np.percentile(time_list, [50, 90, 95, 99])
    print("tp50:{:.4f}ms, tp90:{:.4f}ms,tp95:{:.4f}ms,tp99:{:.4f}ms".format(tp[0] * 1000, tp[1]* 1000, tp[2]* 1000, tp[3]* 1000))
    print("average: {}".format(sum(time_list) / len(time_list)))
    print("qps:{:.4f}".format(len(time_list) / sum(time_list)))

def single_test(url, query_list, num, process_id = 0):
    # query_list:待请求query列表，num请求个数
    print("running process: process-{}".format(process_id))
    time_list = []
    for i in range(num):
        start_time = time.time()
        query = random.choice(query_list)
        requests.post(url, json.dumps({"query": query}))
        end_time = time.time()
        time_list.append(end_time-start_time)
    return time_list


def batch_test(query_list, process_num, request_num):
    # query_list:待请求query列表，process_num进程个数，request_num请求个数(每个进程)
    pool = Pool(processes=process_num)
    process_result = []
    for i in range(process_num):
        process_result.append(pool.apply_async(single_test, args=(query_list, request_num, str(i), )))
        # processes.append(Process(target=single_test, args=(query_list, request_num, str(i), )))
    
    pool.close()
    pool.join()

    time_list = []
    for result in process_result:
        time_list.extend(result.get())
    return time_list   


# response = requests.post("http://127.0.0.1:9090/a", json.dumps({"query": "你好啊1"}))
# print(json.loads(response.text))

# response = requests.post("http://127.0.0.1:9091/b", json.dumps({"query": "你好啊2"}))
# print(json.loads(response.text))

if __name__ == "__main__":
    url = "http://127.0.0.1:9090/searcher"
    print(run_client(url, "什么人不能吃花生")) # 单元测试


    # time_list = [0]
    # time_list = single_test(url, ["你好啊","今天天气怎么样"], 100) # 批量单进程测试
    # cal_time_result(time_list=time_list)
    # time_list = batch_test(url, ["你好啊","今天天气怎么样"], 4, 100) # 多进程压测
    # cal_time_result(time_list=time_list)