import lightning.pytorch as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset



class WaveDatasetAudio(Dataset):
    def __init__(
        self,
        dataset: list[str],
    ) -> None:
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
class WaveDataset(WaveDatasetAudio):
    def __getitem__(self, idx):
        data = self.dataset[idx]
        #TODO


class WaveDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = dataset

    def setup(self, stage: str):
        if stage == "fit":
            (
                train_dataset,
                val_dataset,
                self.train_dataset_name,
                self.val_dataset_name,
            ) = train_test_split(
                self.dataset,
                test_size=self.hparams.val_size,
                random_state=167,
            )
            self.train_dataset = WaveDataset(
                dataset=train_dataset,
            )
            self.val_dataset = WaveDataset(
                dataset=val_dataset,
            )

        if stage == "test":
            self.test_dataset = WaveDataset(
                dataset=self.dataset,
            )   

        else:
            raise AssertionError(
                "'stage' has to be among ['fit', 'test']"
                )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            )
    def val_dataloader(self):
         return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            )
    def test_dataloader(self):
         return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            )