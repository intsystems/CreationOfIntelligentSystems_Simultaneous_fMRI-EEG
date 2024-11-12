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
