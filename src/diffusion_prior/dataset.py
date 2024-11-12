import torch
from torch.utils.data import Dataset, DataLoader


class EmbeddingDataset(Dataset):

    def __init__(self, combined_embeddings, image_embeddings):
        self.combined_embeddings = combined_embeddings
        self.image_embeddings = image_embeddings

    def __len__(self):
        return len(self.combined_embeddings)

    def __getitem__(self, idx):
        return {
            "combined_embedding": self.combined_embeddings[idx],
            "image_embedding": self.image_embeddings[idx]
        }


class EmbeddingDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    def collate_fn(self, batch):
        # Unpack the batch
        combined_embeddings = [item['combined_embedding'] for item in batch]
        image_embeddings = [item['image_embedding'] for item in batch]

        # Stack the embeddings into tensors
        combined_embeddings_tensor = torch.stack(combined_embeddings)
        image_embeddings_tensor = torch.stack(image_embeddings)

        return {
            'combined_embedding': combined_embeddings_tensor,
            'image_embedding': image_embeddings_tensor
        }