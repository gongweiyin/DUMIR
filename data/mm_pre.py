from torch.utils.data import Dataset
import torch
import numpy as np

__all__ = ['MMDataset']

class MMDataset(Dataset):

    def __init__(self, train_data_index, label_ids, text_feats, video_feats, audio_feats):

        self.label_ids = torch.tensor(label_ids)
        self.text_feats = torch.tensor(text_feats)
        self.video_feats = torch.tensor(np.array(video_feats))
        self.audio_feats = torch.tensor(np.array(audio_feats))
        self.size = len(self.text_feats)
        self.sample_index = train_data_index #我添加

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'label_ids': self.label_ids[index],
            'text_feats': self.text_feats[index],
            'video_feats': self.video_feats[index],
            'audio_feats': self.audio_feats[index],
            'sample_index': self.sample_index[index]#我添加
        }
        return sample