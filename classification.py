#!/usr/bin/env python
# coding: utf-8
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

from self_supervised import *
from utils.launch import dist, launch, synchronize
from utils.system_def import *
from defaults import *
import torch
import argparse
import wandb
import json
import sys
import os

import torch.distributed as dist

# os.environ["WANDB_SILENT"] = "true"
global debug


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="The main takes as \
                             argument the parameters dictionary from a json file"
    )
    parser.add_argument(
        "--use-env",
        "--use_env",
        default=False,
        action="store_true",
        help="Use environment variable to pass "
        "'local rank'. For legacy reasons, the default value is False. "
        "If set to True, the script will not pass "
        "--local-rank as argument, and will instead set LOCAL_RANK.",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        required=False,
        default="./run_settings/params_cpg_cdcl_baseline.json",
        help="Give the path of the json file which contains the training parameters",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False, help="Give a valid checkpoint name"
    )
    parser.add_argument(
        "--test", action="store_true", default=False, help="Flag for testing"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Flag for turning on the debug_mode",
    )

    parser.add_argument(
        "--gpu", type=str, required=False, help="The GPU to be used for this run"
    )
    parser.add_argument(
        "--local-rank", type=int, required=False, help="The GPU to be used for this run"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="Used to manipulate the model_name defined in the param file for this run",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        help='Change the "save_dir" in param file',
    )

    parser.add_argument(
        "--knn",
        action="store_true",
        default=False,
        help="Flag for turning on the KNN eval",
    )
    parser.add_argument("--backbone_type", type=str, help="Change backbone")
    parser.add_argument(
        "--pretrained_model_name", type=str, help="pretrained_model_name"
    )

    # semi/self args
    parser.add_argument(
        "--meta", action="store_true", default=False, help="Flag for training with metaDINO"
    )
    parser.add_argument(
        "--byol", action="store_true", default=False, help="Flag for training with BYOL"
    )
    parser.add_argument(
        "--simsiam",
        action="store_true",
        default=False,
        help="Flag for training with SimSiam",
    )
    parser.add_argument(
        "--dino", action="store_true", default=False, help="Flag for training with DINO"
    )
    parser.add_argument(
        "--supervised", action="store_true", default=False, help="Flag for training with inDINO"
    )
    parser.add_argument(
        "--single_cell", action="store_true", default=False, help="Flag for training with DINO"
    )


    # hyperparam settings
    parser.add_argument(
        "--epochs", type=int, required=False, help="number training epochs"
    )
    parser.add_argument(
        "--lr", type=float, required=False, help="lr set for student and teacher"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        required=False,
        help="weight_decay set for student and teacher",
    )

    parser.add_argument(
        "--final_weight_decay",
        type=float,
        required=False,
        help="weight_decay at end epoch",
    )
    parser.add_argument(
        "--barlow_loss_weight",
        type=float,
        required=False,
        help="Importance of barlow loss",
    )
    parser.add_argument(
        "--barlow_lambda_off_diag",
        type=float,
        required=False,
        help="barlow lambda weight, on-off diag",
    )

    parser.add_argument(
        "--only_cross_domain",
        type=bool,
        required=False,
        help="only use cross domain examples",
    )

    parser.add_argument(
        "--freeze_last_for", type=int, required=False, help="freeze last for n epochs"
    )
    parser.add_argument(
        "--center_momentum",
        type=float,
        required=False,
        help="center momentum EMA importance",
    )

    parser.add_argument(
        "--moving_average_decay",
        type=float,
        required=False,
        help="moving average decay start",
    )
    parser.add_argument(
        "--moving_average_decay_end",
        type=float,
        required=False,
        help="moving average decay end",
    )

    parser.add_argument(
        "--bottleneck_dim", type=int, required=False, help="number of bottle neck dims"
    )
    parser.add_argument(
        "--hidden_dim", type=int, required=False, help="num hidden dims"
    )
    parser.add_argument(
        "--projection_size", type=int, required=False, help="projection size"
    )

    parser.add_argument(
        "--norm_last_layer", type=int, required=False, help="norm_last_layer"
    )

    # data setting
    parser.add_argument(
        "--subset_strategy",
        type=str,
        required=False,
        help="Change subset_strategy, only applies if subset is true",
    )
    parser.add_argument(
        "--split_number",
        type=int,
        required=False,
        help="Change split_number, only applies if subset is true",
    )
    parser.add_argument(
        "--data_location", type=str, required=False, help="Update the datapath"
    )

    return parser.parse_args()


def update_params_from_args(params, args):
    if args.gpu:
        prev_gpu = params.system_params.which_GPUs
        params.system_params.which_GPUs = args.gpu  # change the value in-place
        print("Changed GPU for this run from {} to {}".format(prev_gpu, args.gpu))

    if args.model_name:
        prev_model_name = params.training_params.model_name
        params.training_params.model_name = args.model_name
        print(
            "Changed model_name for this run from {} to {}".format(
                prev_model_name, args.model_name
            )
        )

    if args.data_location:
        params["dataset_params"]["data_location"] = args.data_location
        print('Changed data_location to: "{}"'.format(args.data_location))

    if args.save_dir:
        params["training_params"]["save_dir"] = args.save_dir
        print('Changed save_dir to: "{}"'.format(args.save_dir))

    if args.knn:
        params["training_params"]["knn_eval"] = True
        params["model_params"]["freeze_backbone"] = True
        print('Changed knn_eval and freeze_backbone to: "{}"'.format(args.knn))

    if args.backbone_type:
        # update_name = True if args.model_name is None else False
        params["model_params"]["backbone_type"] = args.backbone_type
        print("Changed backbone_type to: {}".format(args.backbone_type))

    if args.epochs:
        params["training_params"]["epochs"] = args.epochs
        print("Changing epoch numbers to {}".format(args.epochs))

    if args.lr:
        params["optimization_params"]["default"]["optimizer"]["params"]["lr"] = args.lr
        print("Changing LR {}".format(args.lr))

    if args.weight_decay:
        params["optimization_params"]["default"]["optimizer"]["params"][
            "weight_decay"
        ] = args.weight_decay
        print("Changing Weight Decay {}".format(args.weight_decay))

    if args.moving_average_decay:
        params["model_params"]["DINO"][
            "moving_average_decay"
        ] = args.moving_average_decay
        print("Changing moving_average_decay {}".format(args.moving_average_decay))

    if args.moving_average_decay_end:
        params["model_params"]["DINO"][
            "moving_average_decay_end"
        ] = args.moving_average_decay_end
        print(
            "Changing moving_average_decay_end {}".format(args.moving_average_decay_end)
        )

    if args.final_weight_decay:
        params["model_params"]["DINO"]["final_weight_decay"] = args.final_weight_decay
        print("Changing final_weight_decay {}".format(args.final_weight_decay))

    if args.barlow_loss_weight:
        params["model_params"]["DINO"]["barlow_loss_weight"] = args.barlow_loss_weight
        print("Changing barlow_loss_weight {}".format(args.barlow_loss_weight))

    if args.only_cross_domain:
        # args.only_cross_domain # change back
        params["model_params"]["DINO"]["only_cross_domain"] = False
        print("Changing only_cross_domain {}".format(args.only_cross_domain))

    if args.projection_size:
        params["model_params"]["DINO"]["projection_size"] = args.projection_size
        print("Changing projection_size {}".format(args.projection_size))

    if args.freeze_last_for:
        params["model_params"]["DINO"]["freeze_last_for"] = args.freeze_last_for
        print("Changing freeze_last_for {}".format(args.freeze_last_for))

    if args.center_momentum:
        params["model_params"]["DINO"]["center_momentum"] = args.center_momentum
        print("Changing center_momentum {}".format(args.center_momentum))

    if args.bottleneck_dim:
        params["model_params"]["DINO"]["bottleneck_dim"] = args.bottleneck_dim
        print("Changing bottleneck_dim {}".format(args.bottleneck_dim))

    if args.barlow_lambda_off_diag:
        params["model_params"]["DINO"][
            "barlow_lambda_off_diag"
        ] = args.barlow_lambda_off_diag
        print("Changing barlow_lambda_off_diag {}".format(args.barlow_lambda_off_diag))

    if args.hidden_dim:
        params["model_params"]["DINO"]["hidden_dim"] = args.hidden_dim
        print("Changing hidden_dim {}".format(args.hidden_dim))

    if args.norm_last_layer is not None:
        params["model_params"]["DINO"]["norm_last_layer"] = args.norm_last_layer
        print("Changing norm_last_layer {}".format(args.norm_last_layer))

    if args.subset_strategy:
        params["dataset_params"]["subset_strategy"] = args.subset_strategy
        print('Changed subset_strategy to: "{}"'.format(args.subset_strategy))

    if args.split_number:
        params["dataset_params"]["split_number"] = args.split_number
        print('Changed split_number to: "{}"'.format(args.split_number))

    if args.pretrained_model_name:
        params["transfer_learning_params"][
            "pretrained_model_name"
        ] = args.pretrained_model_name
        print(
            'Changed pretrained_model_name to: "{}"'.format(args.pretrained_model_name)
        )

    # final step: call the param updaters (schedules etc) if needed
    update_name = True if args.model_name is None else False


def main(parameters, args):

    # dist.init_process_group(backend='nccl')  # Use 'gloo' for CPU or 'nccl' for GPU
    # local_rank = int(os.environ['LOCAL_RANK'])
    # torch.cuda.set_device(local_rank)

    # check self-supervised method
    assert not args.byol * args.simsiam, "BYOL or SimSiam can be on but not both"
    use_momentum = True if args.byol else False

    # define system
    define_system_params(parameters.system_params)

    if args.dino:
        wrapper = DINOWrapper(parameters)
    if args.probability_sampling:
        wrapper = ProbabilitySamplingWrapper(parameters)
    if args.crops:
        wrapper = CropsWrapper(parameters, median=False)
    if args.supervised:
        wrapper = SupervisedWrapper(parameters)
    if args.byol or args.simsiam or args.dino:
        if args.dino:
            wrapper = DINOWrapper(parameters)
    wrapper.instantiate()

    # initialize logger
    if wrapper.is_rank0:
        log_params = wrapper.parameters.log_params
        training_params = wrapper.parameters.training_params
        if wrapper.log_params["run_name"] == "DEFINED_BY_MODEL_NAME":
            log_params["run_name"] = training_params.model_name
            print("model name: ", training_params.model_name)
        if args.debug:
            os.environ["WANDB_MODE"] = "dryrun"
        if not (args.test):
            if parameters.training_params.use_tensorboard:
                print("Using TensorBoard logging")
                summary_writer = SummaryWriter()
            else:
                print("Using WANDB logging")
                wandb.init(
                    project=log_params.project_name,
#                     name=log_params.run_name,
                    config=wrapper.parameters,
                    resume=True if training_params.restore_session else False,
                )
                print("log_params.project_name: ", log_params.project_name)
                print("wandb.run.name: ", wandb.run.name)
                wrapper.parameters["training_params"]["model_name"] = wandb.run.name

    # define trainer
    if args.single_cell:
        trainer = ProbabilitySamplingTrainer(
            wrapper,
            wrapper.parameters.model_params.DINO.moa,
            wrapper.parameters.model_params.DINO.freeze_last_for,
            wrapper.parameters.model_params.DINO.final_weight_decay,
            wrapper.parameters.model_params.DINO.stop_early,
        )
    if args.supervised:
        trainer = SupervisedTrainer(
            wrapper,
            wrapper.parameters.model_params.DINO.freeze_last_for,
            wrapper.parameters.model_params.DINO.final_weight_decay,
            wrapper.parameters.model_params.DINO.stop_early,
        )

    if args.byol or args.simsiam or args.dino or args.indino:
        if args.dino:
            trainer = DINOTrainer(
                wrapper,
                wrapper.parameters.model_params.DINO.freeze_last_for,
                wrapper.parameters.model_params.DINO.final_weight_decay,
                wrapper.parameters.model_params.DINO.stop_early,
            )
        else:
            trainer = BYOLTrainer(wrapper, use_momentum)

    if (parameters.training_params.use_tensorboard) and (wrapper.is_rank0):
        trainer.summary_writer = summary_writer
    elif parameters.training_params.use_tensorboard:
        trainer.summary_writer = None

    if args.test:
        trainer.test(store_embeddings=True)
    else:
        trainer.train()
        trainer.test(store_embeddings=True)

#     dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_arguments()
    parameters = edict(load_params(args))
    update_params_from_args(parameters, args)

#     args.local_rank = int(os.environ["LOCAL_RANK"])

    main(parameters, args)
    
#     try:
#         launch(main, (parameters, args))
#     except Exception as e:
#         if dist.is_initialized():
#             dist.destroy_process_group()
#         raise e
#     finally:
#         if dist.is_initialized():
#             synchronize()
#             dist.destroy_process_group()
