# coding=utf-8
# Filename:    vec_index.py
# Author:      ZENGGUANRONG
# Date:        2023-12-12
# description: 向量召回索引-FAISS

import faiss
from loguru import logger
from src.models.vec_model.vec_model import VectorizeModel

class VecIndex:
    def __init__(self) -> None:
        self.index = ""
    
    def build(self, index_dim):
        description = "HNSW64"
        measure = faiss.METRIC_L2
        self.index = faiss.index_factory(index_dim, description, measure)
    
    def insert(self, vec):
        self.index.add(vec)
    
    def batch_insert(self, vecs):
        self.index.add(vecs)
    
    def load(self, read_path):
        # read_path: XXX.index
        self.index = faiss.read_index(read_path)

    def save(self, save_path):
        # save_path: XXX.index
        faiss.write_index(self.index, save_path)
    
    def search(self, vec, num):
        # id, distance
        return self.index.search(vec, num)