'''
This is the CustomDataset for the conversational english sentences corpus from
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=126
'''
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return data.shape[0]

    def __getitem__(self, idx):
        return data.loc[idx, '원문'], data.loc[idx, '번역문']
    

def get_dataloader(text_file_path, batch_size):
    # Loading Data
    data = pd.read_excel(text_file_path)
    
    # Creating Dataset
    dataset = CustomDataset(data)
    
    # Spliiting into Train/Validation
    train_len = int(len(dataset) * 0.85)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])

    # Creating DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader