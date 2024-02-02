import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from NarrativesDataset import NarrativesDataset

class NarrativesDataModule(pl.LightningDataModule):


    def __init__(self, train_path, val_path, attributes, num_workers, batch_size: int = 16, max_token_len: int = 128, model_name = 'roberta-base'):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.attributes = attributes
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def setup(self, stage = None):
        if stage in (None, "fit"):
            self.train_dataset = NarrativesDataset(self.train_path, attributes = self.attributes, tokenizer = self.tokenizer)
            self.val_dataset = NarrativesDataset(self.val_path, attributes = self.attributes, tokenizer = self.tokenizer, sample = None)
        if stage == 'predict':
            self.val_dataset = NarrativesDataset(self.val_path, attributes = self.attributes, tokenizer = self.tokenizer, sample = None)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False)
    
    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers = self.num_workers, shuffle = False)
