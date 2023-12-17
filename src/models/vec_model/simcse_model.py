import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

class SimcseModel(nn.Module):
    # https://blog.csdn.net/qq_44193969/article/details/126981581
    def __init__(self, pretrained_bert_path, pooling="cls") -> None:
        super(SimcseModel, self).__init__()

        self.pretrained_bert_path = pretrained_bert_path
        self.config = BertConfig.from_pretrained(self.pretrained_bert_path)
        
        self.model = BertModel.from_pretrained(self.pretrained_bert_path, config=self.config)
        self.model.eval()
        
        # self.model = None
        self.pooling = pooling
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        if self.pooling == "cls":
            return out.last_hidden_state[:, 0]
        if self.pooling == "pooler":
            return out.pooler_output
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)
            last = out.hidden_states[-1].transpose(1, 2)
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)