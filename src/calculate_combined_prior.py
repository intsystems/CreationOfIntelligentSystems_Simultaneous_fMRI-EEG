import os
import glob
import torch
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
import torch.distributed as dist

from flow_matching_prior.model import FlowMatchingModel


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
    base_combined_dir = conf.dataloader_kwargs.dataset.combined_dir
    base_name = os.path.basename(base_combined_dir)
    combined_paths = glob.glob(
        os.path.join(base_combined_dir, "**/*.pt"), recursive=True
    )

    model = FlowMatchingModel(**conf.model_kwargs)
    ckpt_path = os.path.join(
        conf.output_dir, "checkpoints", f"checkpoint-{args.steps}.pth"
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.to(device, dtype)
    model.eval()

    pbar = tqdm(
        total=len(combined_paths),
        desc="Calculating combined embeds",
        disable=(rank != 0),
    )

    combined_paths = combined_paths[rank::world_size]
    os.makedirs(args.save_dir, exist_ok=True)

    with torch.no_grad():
        for combined_path in combined_paths:

            try:
                combined_embeds = torch.load(combined_path, map_location="cpu").to(
                    device, dtype
                )

                ts = torch.linspace(0, 1, args.num_inference_steps).to(device)
                for t, dt in zip(ts[:-1], torch.diff(ts)):
                    t = torch.full((1, 1), t).view(-1).to(device, dtype)
                    with torch.no_grad() and torch.autocast("cuda", dtype=dtype):
                        pred_velocity = model(combined_embeds, t)
                    combined_embeds = combined_embeds + dt * pred_velocity

                combined_dir = os.path.dirname(combined_path).replace(
                    base_name, save_name
                )
                os.makedirs(combined_dir, exist_ok=True)
                combined_path = os.path.join(
                    combined_dir, os.path.basename(combined_path)
                )
                torch.save(combined_embeds, combined_path)

            except:
                print(f"Failed to process combined embeds from {combined_path}")
                pass

            if rank == 0:
                pbar.update(world_size)

    if rank == 0:
        pbar.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculating combined embeddings after flow matching prior via FlowMatchingModel"
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
    parser.add_argument(
        "--config_name", type=str, default="flow-matching-prior-fmri-monkeys-29000.yaml"
    )
    parser.add_argument("--steps", type=int, default=13000)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    args = parser.parse_args()
    main(args)
