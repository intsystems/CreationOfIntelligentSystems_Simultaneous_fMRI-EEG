import os
import json
import torch
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
import torch.distributed as dist


from dataset import BrainStimuliDataset, select_random_dimension


def clip_score(combined_embeds, image_embeds):
    combined_embeds = combined_embeds / combined_embeds.norm(p=2, dim=-1, keepdim=True)
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    return (combined_embeds * image_embeds).sum(dim=-1).mean()


class AutoVivification(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


def init_distributed():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def main(args):
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    config_name = "-".join(args.combined_name.split("-")[:-1]) + ".yaml"
    config_path = os.path.join(args.config_dir, config_name)
    conf = OmegaConf.load(config_path)

    if hasattr(conf.dataloader_kwargs.dataset, "combined_dir"):
        base_config_path = os.path.join(args.config_dir, args.base_config_name)
        base_conf = OmegaConf.load(base_config_path)
    else:
        base_conf = conf

    dataset = BrainStimuliDataset(**base_conf.dataloader_kwargs.dataset, mode="none")
    all_indices = list(range(len(dataset)))
    indices = all_indices[rank::world_size]

    pbar = tqdm(
        total=len(all_indices), desc="Calculating CLIP-Score", disable=(rank != 0)
    )

    local_metrics = AutoVivification()
    for key in dataset.data_dict.keys():
        subs = list(dataset.data_dict[key].keys())[1:]
        for sub in subs:
            local_metrics[args.combined_name][key][sub] = []

    for idx in indices:

        try:
            x = dataset[idx]
            combined_dir = os.path.join(
                args.combined_dir,
                args.combined_name,
                x["index"]["key"],
                x["index"]["sub"],
                x["index"]["ses"],
                x["index"]["run"],
            )
            combined_path = os.path.join(
                combined_dir, f"chunk-{x['index']['chunk']}.pt"
            )

            combined_embeds = torch.load(combined_path, map_location="cpu").to(
                device, dtype
            )

            frames = x["frames"].to(device, dtype).unsqueeze(0)
            image_embeds = select_random_dimension(frames)

            with torch.autocast("cuda", dtype=dtype):
                score = float(clip_score(combined_embeds, image_embeds).item())

            local_metrics[args.combined_name][x["index"]["key"]][
                x["index"]["sub"]
            ].append(score)

        except:
            print(f"Failed to load combined embeds from {combined_path}")
            pass

        if rank == 0:
            pbar.update(world_size)

    if rank == 0:
        pbar.close()
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, dict(local_metrics))

    if rank == 0:
        merged_metrics = AutoVivification()
        merged_metrics[args.combined_name] = {
            key: {sub: [] for sub in list(dataset.data_dict[key].keys())[1:]}
            for key in dataset.data_dict.keys()
        }

        for part in gathered:
            for key in part.get(args.combined_name, {}).keys():
                for sub in part[args.combined_name][key].keys():
                    merged_metrics[args.combined_name][key][sub].extend(
                        part[args.combined_name][key][sub]
                    )

        os.makedirs(args.metrics_dir, exist_ok=True)
        save_path = os.path.join(args.metrics_dir, "metrics.json")

        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                prev = json.load(f)
            if args.combined_name not in prev:
                prev[args.combined_name] = {
                    k: {s: [] for s in merged_metrics[args.combined_name][k].keys()}
                    for k in merged_metrics[args.combined_name].keys()
                }
            for key in merged_metrics[args.combined_name].keys():
                if key not in prev[args.combined_name]:
                    prev[args.combined_name][key] = {}
                for sub in merged_metrics[args.combined_name][key].keys():
                    prev[args.combined_name].setdefault(key, {}).setdefault(sub, [])
                    prev[args.combined_name][key][sub].extend(
                        merged_metrics[args.combined_name][key][sub]
                    )
            with open(save_path, "w") as f:
                json.dump(prev, f, indent=4)
        else:
            with open(save_path, "w") as f:
                json.dump(merged_metrics, f, indent=4)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculating CLIP-Score for combined embeddings"
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="/home/jovyan/shares/SR008.fs2/nkiselev/sandbox/visual_stimuli_reconstruction/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/code/data/metrics",
    )
    parser.add_argument(
        "--combined_dir",
        type=str,
        default="/home/jovyan/shares/SR008.fs2/nkiselev/sandbox/visual_stimuli_reconstruction/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/code/data/combined",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="/home/jovyan/shares/SR008.fs2/nkiselev/sandbox/visual_stimuli_reconstruction/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/train_updated/configs",
    )
    parser.add_argument("--combined_name", type=str, default="fmri-monkeys-29000")
    parser.add_argument("--base_config_name", type=str, default="fmri-monkeys.yaml")
    args = parser.parse_args()
    main(args)
