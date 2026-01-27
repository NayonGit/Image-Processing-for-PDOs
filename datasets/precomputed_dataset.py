import torch
from tqdm import tqdm
from torch.utils.data import Dataset

class PrecomputedDataset(Dataset):
    def __init__(self, dataset, model):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.model = model.to(device)
        self.features = []
        self.labels = []
        for x, y in tqdm(self.dataset, desc='Precomputing dataset'):
            self.labels.append(y)
            with torch.no_grad():
                feat = self.model.forward_pass(self.model, x.to(device).unsqueeze(0)).detach().cpu().squeeze()
            self.features.append(feat)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
