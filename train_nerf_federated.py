import argparse
import os
import threading
from distutils.dir_util import copy_tree

import IPython
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from nerf import CfgNode
from train_nerf import create_models, load_dataset, set_seed, train_nerf
from update_compression import create_models_ML, get_parameter_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    configargs = parser.parse_args()

    gpu_count = torch.cuda.device_count()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    (
        USE_CACHED_DATASET,
        train_paths,
        validation_paths,
        images,
        poses,
        render_poses,
        hwf,
        i_split,
        H,
        W,
        focal,
        i_train,
        i_val,
        i_test,
    ) = load_dataset(cfg)

    assert not USE_CACHED_DATASET  # doesn't play nicely with partitioning

    # Partition training dataset
    set_seed(cfg)
    np.random.shuffle(i_train)
    assert cfg.federated.initialise_image_count < len(i_train)
    initialise_ids = i_train[: cfg.federated.initialise_image_count]
    i_train = i_train[cfg.federated.initialise_image_count :]
    partitioned_ids = None  # list of cfg.federated.nodes number of i_train lists
    if cfg.federated.partition == "interleave":  # interleave by angle around z axis
        image_angles = []
        for i, pose in zip(i_train, poses[i_train]):
            coordinate = pose[0:3, 3] / (pose[3, 3] if pose.shape[0] > 3 else 1)
            angle = torch.arctan2(coordinate[1], coordinate[0])
            image_angles.append((i, angle))
        sorted_image_ids_by_angle = [
            i for i, _ in sorted(image_angles, key=lambda ia: ia[1])
        ]
        partitioned_ids = [
            sorted_image_ids_by_angle[x :: cfg.federated.nodes]
            for x in range(cfg.federated.nodes)
        ]
    elif cfg.federated.partition == "separate":
        image_angles = []
        for i, pose in zip(i_train, poses[i_train]):
            coordinate = pose[0:3, 3] / (pose[3, 3] if pose.shape[0] > 3 else 1)
            angle = torch.arctan2(coordinate[1], coordinate[0])
            image_angles.append((i, angle))
        sorted_image_ids_by_angle = [
            i for i, _ in sorted(image_angles, key=lambda ia: ia[1])
        ]
        partitioned_ids = [
            list(l)
            for l in np.array_split(sorted_image_ids_by_angle, cfg.federated.nodes)
        ]
    else:
        raise ValueError()

    print("Initialise IDs: ", initialise_ids)
    print("Training partition IDs: ", partitioned_ids)

    # Setup global logging and copy config
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=False)
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    # Create and train the initialisation NeRF
    device = torch.device("cuda:0")
    set_seed(
        cfg
    )  # by resetting seeds before creating each model, they have the same initial weights
    (
        init_encode_position_fn,
        init_encode_direction_fn,
        init_model_coarse,
        init_model_fine,
        init_trainable_parameters,
        init_optimizer,
    ) = create_models(cfg, device)
    if isinstance(cfg.federated.train_initial, bool) and cfg.federated.train_initial:
        individual_logdir = os.path.join(logdir, f"node_initial")
        writer = SummaryWriter(individual_logdir)
        os.makedirs(individual_logdir, exist_ok=True)
        print("############# TRAINING INITIAL NERF")
        train_nerf(
            device,
            cfg,
            0,
            cfg.experiment.train_iters,
            USE_CACHED_DATASET,
            init_model_coarse,
            init_model_fine,
            init_optimizer,
            writer,
            individual_logdir,
            train_paths,
            validation_paths,
            init_encode_position_fn,
            init_encode_direction_fn,
            initialise_ids,
            i_val,
            images,
            poses,
            H,
            W,
            focal,
        )
        print("################ INITIAL NERF TRAINED")
        torch.cuda.empty_cache()

    elif isinstance(
        cfg.federated.train_initial, str
    ) and cfg.federated.train_initial.startswith("fromexp:"):
        load_from_exp_name = cfg.federated.train_initial[8:]
        copy_tree(
            os.path.join(cfg.experiment.logdir, load_from_exp_name, "node_initial"),
            os.path.join(cfg.experiment.logdir, cfg.experiment.id, "node_initial"),
        )
        ckpt = torch.load(
            os.path.join(
                cfg.experiment.logdir,
                cfg.experiment.id,
                "node_initial",
                f"checkpoint{str((cfg.experiment.train_iters-1)).zfill(5)}.ckpt",
            ),
            map_location=device,
        )
        init_model_coarse.load_state_dict(ckpt["model_coarse_state_dict"])
        if init_model_fine:
            init_model_fine.load_state_dict(ckpt["model_fine_state_dict"])
        torch.cuda.empty_cache()
    else:
        raise ValueError()

    # Create and train the control NeRF
    if isinstance(cfg.federated.train_control, bool) and cfg.federated.train_control:
        set_seed(
            cfg
        )  # by resetting seeds before creating each model, they have the same initial weights
        (
            control_encode_position_fn,
            control_encode_direction_fn,
            control_model_coarse,
            control_model_fine,
            control_trainable_parameters,
            control_optimizer,
        ) = create_models(cfg, device)
        control_model_coarse.load_state_dict(init_model_coarse.state_dict())
        if control_model_fine is not None:
            control_model_fine.load_state_dict(init_model_fine.state_dict())
        individual_logdir = os.path.join(logdir, f"node_control")
        writer = SummaryWriter(individual_logdir)
        os.makedirs(individual_logdir, exist_ok=True)
        print("############# TRAINING CONTROL NERF")
        train_nerf(
            device,
            cfg,
            0,
            cfg.experiment.train_iters,
            USE_CACHED_DATASET,
            control_model_coarse,
            control_model_fine,
            control_optimizer,
            writer,
            individual_logdir,
            train_paths,
            validation_paths,
            control_encode_position_fn,
            control_encode_direction_fn,
            i_train,  # all the remaning non-initialisation data together :)
            i_val,
            images,
            poses,
            H,
            W,
            focal,
        )
        print("################ CONTROL NERF TRAINED")
        torch.cuda.empty_cache()
    elif isinstance(
        cfg.federated.train_control, str
    ) and cfg.federated.train_control.startswith("fromexp:"):
        load_from_exp_name = cfg.federated.train_control[8:]
        copy_tree(
            os.path.join(cfg.experiment.logdir, load_from_exp_name, "node_control"),
            os.path.join(cfg.experiment.logdir, cfg.experiment.id, "node_control"),
        )
        torch.cuda.empty_cache()
    else:
        raise ValueError()

    # Create the separate NeRFs
    nerfs = []
    for i in range(cfg.federated.nodes):
        device = torch.device("cuda", i % gpu_count)
        set_seed(
            cfg
        )  # by resetting seeds before creating each model, they have the same initial weights
        (
            encode_position_fn,
            encode_direction_fn,
            model_coarse,
            model_fine,
            trainable_parameters,
            optimizer,
        ) = create_models(cfg, device)
        # Setup nerf from the initially trained model
        if cfg.federated.compress_method == "none":
            model_coarse.load_state_dict(init_model_coarse.state_dict().to(device))
            if model_fine is not None:
                model_fine.load_state_dict(init_model_fine.state_dict().to(device))
        elif cfg.federated.compress_method == "ML":
            (
                model_coarse,
                model_fine,
                trainable_parameters,
                optimizer,
            ) = create_models_ML(init_model_coarse, init_model_fine, cfg)
            model_coarse = model_coarse.to(device)
            model_fine = model_fine.to(device)
            if i == 0:
                print(
                    f"###### COMPRESSION FROM INITIAL {get_parameter_count(init_model_coarse)} TO {get_parameter_count(model_coarse)} PARAMETERS"
                )
        else:
            raise ValueError

        individual_logdir = os.path.join(logdir, f"node{i}")
        writer = SummaryWriter(individual_logdir)
        os.makedirs(individual_logdir, exist_ok=True)
        nerfs.append(
            dict(
                i_train=partitioned_ids[i],
                device=device,
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                model_coarse=model_coarse,
                model_fine=model_fine,
                trainable_parameters=trainable_parameters,
                optimizer=optimizer,
                logdir=individual_logdir,
                writer=writer,
            )
        )

    # Federated training loop
    torch.cuda.empty_cache()
    start_iter = 0
    for iter_upto in trange(
        cfg.federated.merge_every,
        cfg.experiment.train_iters + 1,
        cfg.federated.merge_every,
        desc="Overall progress",
    ):
        # Train up to next step
        training_threads = []
        print(f"########### TRAINING FROM {start_iter} TO {iter_upto}")
        coarse_combined_state_dict = None  # initialise here to accumulate over all "batches" of NeRFs that can run in the GPUs at once
        fine_combined_state_dict = None
        combine_device = torch.device("cpu")
        # Train by batch that can run on the provided GPU count
        for current_nerfs in tqdm(
            [nerfs[x : x + gpu_count] for x in range(0, len(nerfs), gpu_count)],
            desc="GPU set batches",
        ):
            for nerf in current_nerfs:
                training_threads.append(
                    threading.Thread(
                        target=train_nerf,
                        args=(
                            nerf["device"],
                            cfg,
                            start_iter,
                            iter_upto,
                            USE_CACHED_DATASET,
                            nerf["model_coarse"],
                            nerf["model_fine"],
                            nerf["optimizer"],
                            nerf["writer"],
                            nerf["logdir"],
                            train_paths,
                            validation_paths,
                            nerf["encode_position_fn"],
                            nerf["encode_direction_fn"],
                            nerf["i_train"],
                            i_val,
                            images,
                            poses,
                            H,
                            W,
                            focal,
                        ),
                    )
                )
                training_threads[-1].start()

            for thread in training_threads:
                thread.join()
            torch.cuda.empty_cache()

            # Combine the current weights (leaving individual optimizer states as they are)
            #  start with the weighted weights of the first network if necessary
            reinitialise_state_dict = False
            if coarse_combined_state_dict is None:
                reinitialise_state_dict = True
                coarse_combined_state_dict = {
                    k: len(nerfs[0]["i_train"]) * t.to(combine_device)
                    for k, t in nerfs[0]["model_coarse"].state_dict().items()
                }
                fine_combined_state_dict = {
                    k: len(nerfs[0]["i_train"]) * t.to(combine_device)
                    for k, t in nerfs[0]["model_fine"].state_dict().items()
                }
            #  now add the weights of the remaining networks
            for nerf in current_nerfs[1 if reinitialise_state_dict else 0 :]:
                coarse_combined_state_dict = {
                    k: coarse_combined_state_dict[k]
                    + len(nerf["i_train"]) * t.to(combine_device)
                    for k, t in nerf["model_coarse"].state_dict().items()
                }
                fine_combined_state_dict = {
                    k: fine_combined_state_dict[k]
                    + len(nerf["i_train"]) * t.to(combine_device)
                    for k, t in nerf["model_fine"].state_dict().items()
                }

        # Finally, divide by total weights now that all GPU set batches are done
        total_weight = len(i_train)
        coarse_combined_state_dict = {
            k: t / total_weight for k, t in coarse_combined_state_dict.items()
        }
        fine_combined_state_dict = {
            k: t / total_weight for k, t in fine_combined_state_dict.items()
        }

        # Now update all NeRFs with the combined weights and also save a checkpoint
        for nerf in nerfs:
            nerf["model_coarse"].load_state_dict(coarse_combined_state_dict)
            nerf["model_fine"].load_state_dict(fine_combined_state_dict)
        os.makedirs(os.path.join(logdir, "nodes_combined"), exist_ok=True)
        torch.save(
            {
                "model_coarse_state_dict": coarse_combined_state_dict,
                "model_fine_state_dict": fine_combined_state_dict,
                "iter": iter_upto - 1,
                "optimizer_state_dicts": [
                    nerf["optimizer"].state_dict() for nerf in nerfs
                ],
                "partitioned_ids": partitioned_ids,
            },
            os.path.join(
                logdir, "nodes_combined", f"checkpoint{str(iter_upto-1).zfill(5)}.ckpt"
            ),
        )
        print("############ SAVED COMBINED CHECKPOINT")
        torch.cuda.empty_cache()
        start_iter = iter_upto
