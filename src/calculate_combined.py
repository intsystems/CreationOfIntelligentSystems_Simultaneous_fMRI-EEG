import os
import torch
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
import torch.distributed as dist

from dataset import BrainStimuliDataset
from brain_encoder import BrainEncoder, fMRIBrainEncoder, EEGBrainEncoder


def init_distributed():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def main(args):
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16
    torch.cuda.set_device(device)

    config_path = os.path.join(args.config_dir, args.config_name)
    save_name = os.path.splitext(args.config_name)[0] + f"-{args.steps}"
    conf = OmegaConf.load(config_path)

    dataset = BrainStimuliDataset(**conf.dataloader_kwargs.dataset, mode=conf.mode)

    if conf.mode == "fmri":
        Model = fMRIBrainEncoder
    elif conf.mode == "eeg":
        Model = EEGBrainEncoder
    elif conf.mode == "both":
        Model = BrainEncoder
    else:
        raise ValueError(f"Unknown mode: {conf.mode}")

    model = Model(**conf.model_kwargs, num_subs=dataset.num_subs)
    ckpt_path = os.path.join(
        conf.output_dir, "checkpoints", f"checkpoint-{args.steps}.pth"
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.to(device, dtype)
    model.eval()

    all_indices = list(range(len(dataset)))
    shard = all_indices[rank::world_size]

    pbar = tqdm(
        total=len(all_indices), desc="Calculating combined embeds", disable=(rank != 0)
    )

    os.makedirs(args.save_dir, exist_ok=True)

    with torch.no_grad():
        for idx in shard:
            x = dataset[idx]
            sub_id = torch.tensor([x["id"]], device=device)

            fmri_embeds = (
                x["fmri"].unsqueeze(0).to(device, dtype)
                if x["fmri"] is not None
                else None
            )
            eeg_embeds = (
                x["eeg"].unsqueeze(0).to(device, dtype)
                if x["eeg"] is not None
                else None
            )

            with torch.no_grad() and torch.autocast("cuda", dtype=dtype):
                combined_embeds = model(
                    sub_ids=sub_id, batch_eeg=eeg_embeds, batch_fmri=fmri_embeds
                ).squeeze(0)

            combined_dir = os.path.join(
                args.save_dir,
                save_name,
                x["index"]["key"],
                x["index"]["sub"],
                x["index"]["ses"],
                x["index"]["run"],
            )
            os.makedirs(combined_dir, exist_ok=True)
            combined_path = os.path.join(
                combined_dir, f"chunk-{x['index']['chunk']}.pt"
            )
            torch.save(combined_embeds, combined_path)

            if rank == 0:
                pbar.update(world_size)

    if rank == 0:
        pbar.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculating combined embeddings via BrainEncoder"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/jovyan/shares/SR008.fs2/nkiselev/sandbox/visual_stimuli_reconstruction/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/code/data/combined",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="/home/jovyan/shares/SR008.fs2/nkiselev/sandbox/visual_stimuli_reconstruction/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/train_updated/configs",
    )
    parser.add_argument("--config_name", type=str, default="fmri-monkeys.yaml")
    parser.add_argument("--steps", type=int, default=44000)
    args = parser.parse_args()
    main(args)
