"""Minimal Octo inference example (no install steps).

Usage:
    python examples/inference_pretrained.py                    # quick single-image demo
    python examples/inference_pretrained.py --full-trajectory  # run on a Bridge dataset episode
    python examples/inference_pretrained.py --benchmark        # measure inference latency
"""
import argparse
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import jax
import numpy as np
from PIL import Image
import requests

from octo.model.octo_model import OctoModel


def single_image_inference(model):
    """Load one BridgeV2 image, run a text-conditioned action prediction."""
    IMAGE_URL = (
        "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/"
        "bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/"
        "2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
    )
    img = np.array(
        Image.open(requests.get(IMAGE_URL, stream=True).raw).resize((256, 256))
    )

    # add batch + time horizon dims  ->  (1, 1, 256, 256, 3)
    observation = {
        "image_primary": img[np.newaxis, np.newaxis, ...],
        "timestep_pad_mask": np.array([[True]]),
    }
    task = model.create_tasks(texts=["pick up the fork"])

    t0 = time.perf_counter()
    action = model.sample_actions(
        observation,
        task,
        unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"],
        rng=jax.random.PRNGKey(0),
    )
    jax.block_until_ready(action)
    elapsed = (time.perf_counter() - t0) * 1000

    print("Predicted action (batch, action_chunk, action_dim):")
    print(action)
    print(f"\nInference time: {elapsed:.2f} ms (includes JIT compilation on first call)")


def benchmark_inference(model, n_warmup=5, n_iters=50):
    """Measure inference latency with perf_counter."""
    IMAGE_URL = (
        "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/"
        "bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/"
        "2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
    )
    img = np.array(
        Image.open(requests.get(IMAGE_URL, stream=True).raw).resize((256, 256))
    )
    observation = {
        "image_primary": img[np.newaxis, np.newaxis, ...],
        "timestep_pad_mask": np.array([[True]]),
    }
    task = model.create_tasks(texts=["pick up the fork"])
    stats = model.dataset_statistics["bridge_dataset"]["action"]

    # warmup — first calls include JIT compilation
    print(f"Warming up ({n_warmup} iterations) ...")
    for i in range(n_warmup):
        model.sample_actions(
            observation, task, unnormalization_statistics=stats,
            rng=jax.random.PRNGKey(i),
        )
    # wait for async dispatch to finish
    jax.block_until_ready(
        model.sample_actions(
            observation, task, unnormalization_statistics=stats,
            rng=jax.random.PRNGKey(0),
        )
    )

    # timed run
    print(f"Benchmarking ({n_iters} iterations) ...")
    times = []
    for i in range(n_iters):
        t0 = time.perf_counter()
        action = model.sample_actions(
            observation, task, unnormalization_statistics=stats,
            rng=jax.random.PRNGKey(i),
        )
        jax.block_until_ready(action)
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # to ms
    print(f"\n--- Inference Benchmark ({n_iters} iters) ---")
    print(f"  Mean:   {times.mean():.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Std:    {times.std():.2f} ms")
    print(f"  Min:    {times.min():.2f} ms")
    print(f"  Max:    {times.max():.2f} ms")
    print(f"  Throughput: {1000 / times.mean():.1f} it/sec")


def full_trajectory_inference(model):
    """Load one Bridge episode from TFDS, predict actions, and plot vs ground truth."""
    import cv2
    import matplotlib.pyplot as plt
    import tensorflow_datasets as tfds
    import tqdm

    # --- load data ---
    builder = tfds.builder_from_directory(
        builder_dir="gs://gresearch/robotics/bridge/0.1.0/"
    )
    ds = builder.as_dataset(split="train[:1]")
    episode = next(iter(ds))
    steps = list(episode["steps"])
    images = [
        cv2.resize(np.array(step["observation"]["image"]), (256, 256))
        for step in steps
    ]
    language_instruction = (
        steps[0]["observation"]["natural_language_instruction"].numpy().decode()
    )
    print(f"Instruction: {language_instruction}")

    # --- inference loop ---
    WINDOW_SIZE = 2
    task = model.create_tasks(texts=[language_instruction])

    pred_actions, true_actions = [], []
    for step_idx in tqdm.trange(len(images) - (WINDOW_SIZE - 1)):
        input_images = np.stack(images[step_idx : step_idx + WINDOW_SIZE])[None]
        observation = {
            "image_primary": input_images,
            "timestep_pad_mask": np.full(
                (1, input_images.shape[1]), True, dtype=bool
            ),
        }
        actions = model.sample_actions(
            observation,
            task,
            unnormalization_statistics=model.dataset_statistics["bridge_dataset"][
                "action"
            ],
            rng=jax.random.PRNGKey(0),
        )
        pred_actions.append(actions[0])

        fw = step_idx + WINDOW_SIZE - 1
        true_actions.append(
            np.concatenate(
                (
                    steps[fw]["action"]["world_vector"],
                    steps[fw]["action"]["rotation_delta"],
                    np.array(steps[fw]["action"]["open_gripper"]).astype(np.float32)[
                        None
                    ],
                ),
                axis=-1,
            )
        )

    # --- plot ---
    ACTION_DIM_LABELS = ["x", "y", "z", "yaw", "pitch", "roll", "grasp"]
    pred_actions = np.array(pred_actions).squeeze()
    true_actions = np.array(true_actions).squeeze()

    fig, axs = plt.subplots(1, len(ACTION_DIM_LABELS), figsize=(45, 5))
    for i, label in enumerate(ACTION_DIM_LABELS):
        axs[i].plot(pred_actions[:, 0, i], label="predicted")
        axs[i].plot(true_actions[:, i], label="ground truth")
        axs[i].set_title(label)
        axs[i].set_xlabel("Timestep")
    plt.legend()
    plt.tight_layout()
    plt.savefig("action_comparison.png", dpi=100)
    print("Saved plot to action_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-trajectory",
        action="store_true",
        help="Run inference on a full Bridge dataset episode (requires GCS access)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Measure inference latency (excludes JIT warmup)",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=50,
        help="Number of timed iterations for --benchmark (default: 50)",
    )
    parser.add_argument(
        "--checkpoint",
        default="hf://rail-berkeley/octo-small-1.5",
        help="Model checkpoint path or HuggingFace ID",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint} ...")
    model = OctoModel.load_pretrained(args.checkpoint)

    if args.benchmark:
        benchmark_inference(model, n_iters=args.benchmark_iters)
    elif args.full_trajectory:
        full_trajectory_inference(model)
    else:
        single_image_inference(model)
