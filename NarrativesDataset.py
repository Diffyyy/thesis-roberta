import pandas as pd
from torch.utils.data import Dataset
import torch

class NarrativesDataset(Dataset):
    def __init__(self, data_path, tokenizer, attributes, max_token_len: int = 128, sample = 5000):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.attributes = attributes
        self.max_token_len = max_token_len
        self.sample = sample
        self.prepare_data()

    def prepare_data(self):
        self.data = pd.read_csv(self.data_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data.iloc[index]
        narrative = str(item['Narrative'])
        attributes = torch.FloatTensor(item[self.attributes])
        tokens = self.tokenizer.encode_plus(narrative, 
                                            add_special_tokens = True,
                                            return_tensors = 'pt',
                                            truncation = True,
                                            padding = 'max_length',
                                            max_length = self.max_token_len,
                                            return_attention_mask = True )
        
        return{'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': attributes}