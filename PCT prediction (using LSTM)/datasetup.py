import numpy as np
from torch.utils.data import Dataset
import torch
from itertools import zip_longest


# ......
class ModelDataSetup(Dataset):

    # ...
    def __init__(self, ds:dict, device:torch.device):
        
        super().__init__()
        self.ds = ds
        
        # ... trains with raw input
        if False:
            pct_beyond_simulation = 100
            tgt_max_len = 0
            for i in range(len(self.ds)):
                tgt_len = len(self.ds[i]["tgt"])
                if tgt_len > tgt_max_len:
                    tgt_max_len = tgt_len
            
            for i in range(len(self.ds)):
                arr = self.ds[i]
                padded_tgt = np.concatenate(
                    (arr["tgt"], np.array([pct_beyond_simulation] * (tgt_max_len - len(arr["tgt"])))))
                self.ds[i]["tgt"] = padded_tgt
        
        # ... trains via scaled input
        if True:
            pass

        self.device = device
    
    # ...
    def __len__(self):

        return len(self.ds)
    
    # ...
    def __getitem__(self, index):

        src = torch.tensor(self.ds[index]["src"], dtype=torch.float32).to(self.device)
        tgt = torch.tensor(self.ds[index]["tgt"], dtype=torch.float32).to(self.device)

        return {
            "src": src,
            "tgt": tgt,
        }