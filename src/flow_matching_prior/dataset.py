import os
import json
import torch
from torch.utils.data import Dataset, DataLoader


def load_json_data(json_path):
    with open(json_path, "r") as file:
        data_dict = json.load(file)
    return data_dict


def prepare_embedding_paths(json_path, combined_dir):

    data_dict = load_json_data(json_path)
    embedding_paths = []

    for key in data_dict.keys():
        for sub in list(data_dict[key].keys())[1:]:
            for ses in data_dict[key][sub].keys():
                for run in data_dict[key][sub][ses].keys():
                    for chunk_idx in data_dict[key][sub][ses][run]["chunks"].keys():

                        # make path to combined embeds
                        combined_path = os.path.join(
                            combined_dir, key, sub, ses, run, f"chunk-{chunk_idx}.pt"
                        )

                        # check if the combined embeds exists
                        if not os.path.exists(combined_path):
                            continue

                        # get time indices for frames
                        time_indices = data_dict[key][sub][ses][run]["chunks"][
                            chunk_idx
                        ]["frames"]

                        # make path to image embeds
                        frame_paths = [
                            os.path.join(
                                data_dict[key]["frames_dir"],
                                f"frame_{frame_idx:04d}.pt",
                            )
                            for frame_idx in range(
                                time_indices["start_idx"], time_indices["end_idx"] + 1
                            )
                        ]

                        # append paths to list
                        embedding_paths.append(
                            {"combined": combined_path, "image": frame_paths}
                        )

    return embedding_paths


class EmbeddingDataset(Dataset):

    def __init__(self, json_path, combined_dir):
        self.embedding_paths = prepare_embedding_paths(json_path, combined_dir)

    def __len__(self):
        return len(self.embedding_paths)

    def __getitem__(self, idx):

        # load combined embeds
        combined_embedding = torch.load(
            self.embedding_paths[idx]["combined"], map_location="cpu", weights_only=True
        )

        # load image embeds
        image_embedding = torch.stack(
            list(
                map(
                    lambda x: torch.load(x, map_location="cpu", weights_only=True),
                    self.embedding_paths[idx]["image"],
                )
            )
        )

        return {
            "combined_embedding": combined_embedding,
            "image_embedding": image_embedding,
        }


class EmbeddingDataLoader(DataLoader):
    def __init__(
        self, dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def collate_fn(self, batch):
        # Unpack the batch
        combined_embeddings = [item["combined_embedding"] for item in batch]
        image_embeddings = [item["image_embedding"] for item in batch]

        # Stack the embeddings into tensors
        combined_embeddings_tensor = torch.stack(combined_embeddings)
        image_embeddings_tensor = torch.stack(image_embeddings)

        return {
            "combined_embedding": combined_embeddings_tensor,
            "image_embedding": image_embeddings_tensor,
        }
