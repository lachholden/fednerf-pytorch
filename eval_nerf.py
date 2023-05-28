import argparse
import csv
import os
import time

import imageio
import IPython
import lpips
import numpy as np
import torch
import torchvision
import yaml
from pytorch_msssim import ssim
from tqdm import tqdm

import update_compression
from nerf import (
    CfgNode,
    get_embedding_function,
    get_ray_bundle,
    load_blender_data,
    load_llff_data,
    models,
    run_one_iter_of_nerf,
)
from nerf.nerf_helpers import img2mse, mse2psnr


def cast_to_image(tensor, dataset_type):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Convert to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    return img
    # # Map back to shape (3, H, W), as tensorboard needs channels first.
    # return np.moveaxis(img, [-1], [0])


def cast_to_disparity_image(tensor):
    img = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    img = img.clamp(0, 1) * 255
    return img.detach().cpu().numpy().astype(np.uint8)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint / pre-trained model to evaluate.",
    )
    parser.add_argument(
        "--csvfile",
        type=str,
        required=True,
        help="The path of the file to save the test statistics",
    )
    parser.add_argument(
        "--checkpoint-type",
        type=str,
        required=True,
        choices=["federated", "federated_cumiter", "single_node", "control", "initial"],
        help="The type of checkpoint",
    )
    parser.add_argument(
        "--savedir", type=str, help="Save images to this directory, if specified."
    )
    parser.add_argument(
        "--save-disparity-image", action="store_true", help="Save disparity images too."
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    images, poses, render_poses, hwf = None, None, None, None
    i_train, i_val, i_test = None, None, None
    if cfg.dataset.type.lower() == "blender":
        # Load blender dataset
        images, poses, render_poses, hwf, i_split = load_blender_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
        )
        i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
        if cfg.nerf.train.white_background:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    elif cfg.dataset.type.lower() == "llff":
        # Load LLFF dataset
        images, poses, bds, render_poses, i_test = load_llff_data(
            cfg.dataset.basedir,
            factor=cfg.dataset.downsample_factor,
        )
        hwf = poses[0, :3, -1]
        H, W, focal = hwf
        hwf = [int(H), int(W), focal]
        render_poses = torch.from_numpy(render_poses)

        # LLFF only uses a single holdout test image, but make it a list to align with
        # the blender convention
        i_test = [int(i_test)]
        print(i_test)

    # Device on which to run.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
    )
    model_coarse.to(device)

    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
        )
        model_fine.to(device)

    checkpoint = torch.load(configargs.checkpoint, map_location=device)

    # If federated compression was used for the experiment, reparameterise models
    if configargs.checkpoint_type in ["federated", "single_node", "federated_cumiter"]:
        if cfg.federated.compress_method == "ML":
            model_coarse = update_compression.reparameterise_model_ML_from_state_dict(
                model_coarse, checkpoint["model_coarse_state_dict"]
            )
            if model_fine is not None:
                model_fine = update_compression.reparameterise_model_ML_from_state_dict(
                    model_fine, checkpoint["model_fine_state_dict"]
                )

    model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
    if checkpoint["model_fine_state_dict"]:
        try:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        except:
            print(
                "The checkpoint has a fine-level model, but it could "
                "not be loaded (possibly due to a mismatched config file."
            )

    # All the previous state dict and reparameterisation stuff is messy as, so just do
    # this to 100% make sure
    model_coarse.to(device)
    model_fine.to(device)

    if "height" in checkpoint.keys():
        hwf[0] = checkpoint["height"]
    if "width" in checkpoint.keys():
        hwf[1] = checkpoint["width"]
    if "focal_length" in checkpoint.keys():
        hwf[2] = checkpoint["focal_length"]

    model_coarse.eval()
    if model_fine:
        model_fine.eval()

    render_poses = render_poses.float().to(device)

    # Create directory to save images to.
    if configargs.savedir is not None:
        os.makedirs(configargs.savedir, exist_ok=True)
        if configargs.save_disparity_image:
            os.makedirs(os.path.join(configargs.savedir, "disparity"), exist_ok=True)

    csv_f = open(configargs.csvfile, "w")
    csv_writer = csv.writer(csv_f)
    csv_header = [
        "exp_id",
        "config_file",
        "ckpt",
        "ckpt_type",
        "dataset",
        "param_count_coarse",
        "param_count_fine",
        "buffer_count_coarse",
        "buffer_count_fine",
        "weight_count_coarse",
        "weight_count_fine",
        "bias_count_coarse",
        "bias_count_fine",
        "train_iters",
        "initialise_image_count",
        "node_count",
        "partition_method",
        "merge_every",
        "compress_rank",
        "compress_rank_variance_proportion",
        "compress_method",
        "img_i",
        "loss",
        "psnr",
        "ssim",
        "lpips",
    ]
    csv_writer.writerow(csv_header)

    # Evaluation loop
    times_per_image = []
    # for i, pose in enumerate(tqdm(render_poses)):
    loss_fn_vgg = lpips.LPIPS(net="vgg")
    for i in tqdm(i_test):
        pose = poses[i]
        target = torch.Tensor(images[i]).to(device)[:, :, :3]  # discard 4th channel
        start = time.time()
        rgb = None, None
        disp = None, None

        # Render at pose
        with torch.no_grad():
            pose = pose[:3, :4]
            if isinstance(pose, np.ndarray):
                pose = torch.from_numpy(pose).to(device)
            ray_origins, ray_directions = get_ray_bundle(hwf[0], hwf[1], hwf[2], pose)
            rgb_coarse, disp_coarse, _, rgb_fine, disp_fine, _ = run_one_iter_of_nerf(
                hwf[0],
                hwf[1],
                hwf[2],
                model_coarse,
                model_fine,
                ray_origins.to(device),
                ray_directions.to(device),
                cfg,
                mode="validation",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )
            rgb_save = rgb_fine if rgb_fine is not None else rgb_coarse
            if configargs.save_disparity_image:
                disp = disp_fine if disp_fine is not None else disp_coarse
        times_per_image.append(time.time() - start)

        # Calculate comparison metrics for image
        #  loss and PSNR
        rgb = rgb_save.clone().to("cpu")
        target = target.to("cpu")
        img_loss = img2mse(rgb, target)
        loss = img_loss
        psnr = mse2psnr(img_loss)

        # SSIM
        #   need to permute axes from H,W,D -> N,D,H,W
        rgb = torch.permute(rgb, (2, 0, 1))
        rgb = torch.unsqueeze(rgb, 0)
        target = torch.permute(target, (2, 0, 1))
        target = torch.unsqueeze(target, 0)
        ssim_val = ssim(rgb, target, data_range=1)

        # LPIPS
        #  uses same axis configuration as SSIM
        #  normalise to [-1,1] from [0,1]
        target = 2 * target - 1
        rgb = 2 * rgb - 1
        lpips_loss = loss_fn_vgg(rgb, target)

        print(
            f"loss: {loss.item()}, psnr: {psnr}, ssim: {ssim_val.item()}, lpips: {lpips_loss.item()}"
        )

        # Write CSV statistics
        csv_row = [
            cfg.experiment.logdir + "/" + cfg.experiment.id,
            configargs.config,
            configargs.checkpoint,
            configargs.checkpoint_type,
            cfg.dataset.basedir,
            update_compression.get_parameter_count(model_coarse),
            update_compression.get_parameter_count(model_fine) if model_fine else None,
            update_compression.get_buffer_count(model_coarse),
            update_compression.get_buffer_count(model_fine),
            update_compression.get_weight_count(model_coarse),
            update_compression.get_weight_count(model_fine),
            update_compression.get_bias_count(model_coarse),
            update_compression.get_bias_count(model_fine),
            cfg.experiment.train_iters,
            cfg.federated.initialise_image_count,
            cfg.federated.nodes,
            cfg.federated.partition,
            cfg.federated.merge_every,
            cfg.federated.compress_rank
            if cfg.federated.compress_rank != "none"
            else None,
            cfg.federated.compress_rank_variance_proportion
            if cfg.federated.compress_rank_variance_proportion != "none"
            else None,
            cfg.federated.compress_method,
            i,
            loss.item(),
            psnr,
            ssim_val.item(),
            lpips_loss.item(),
        ]
        csv_writer.writerow(csv_row)

        # Save image and disparity if required
        if configargs.savedir:
            savefile = os.path.join(configargs.savedir, f"{i:04d}.png")
            imageio.imwrite(
                savefile, cast_to_image(rgb_save[..., :3], cfg.dataset.type.lower())
            )
            if configargs.save_disparity_image:
                savefile = os.path.join(configargs.savedir, "disparity", f"{i:04d}.png")
                imageio.imwrite(savefile, cast_to_disparity_image(disp))
        tqdm.write(f"Avg time per image: {sum(times_per_image) / (i + 1)}")

    csv_f.close()


if __name__ == "__main__":
    main()
