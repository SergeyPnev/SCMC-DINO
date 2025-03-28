#!/usr/bin/env python
# coding: utf-8

import argparse
from defaults import *
from helpfuns import *
import sys

sys.path.append("..")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="The main takes as \
                             argument the parameters dictionary from a json file"
    )
    parser.add_argument(
        "--params_path",
        type=str,
        required=False,
        default="../params.json",
        help="Give the path of the json file which contains the training parameters",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        required=False,
        default=256,
        help="Size of the updated images",
    )
    parser.add_argument(
        "--num_workers", type=int, required=False, default=18, help="Number of workers"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Flag for doing resising in all data (Default: False -- Only for datasets which use=True)",
    )
    return parser.parse_args()


def compute_stats(dataset=None, dataloader=None, img_size=None):
    from tqdm import tqdm
    from torch.utils.data import Dataset, DataLoader
    from torchvision.transforms import Compose, Resize, ToTensor

    if dataset is None and dataloader is None:
        raise ValueError("Please give as argumets a dataloader or a dataset")

    dataloader_params = {
        "batch_size": 100,
        "num_workers": 12,
        "prefetch_factor": 1,
        "shuffle": False,
        "pin_memory": False,
        "drop_last": False,
        "persistent_workers": False,
    }
    if dataloader is not None:
        print("Using given dataloader to compute stats")
        pass
    elif dataset is not None:
        if img_size is not None:
            dataset.transform = Compose([Resize(img_size), ToTensor()])
        else:
            dataset.transform = Compose([ToTensor()])
        dataloader = DataLoader(dataset, **dataloader_params)
    else:
        raise ValueError("Input args not understood or ill-defiined")

    channels = dataloader.dataset.img_channels
    x_tot = np.zeros(channels)
    x2_tot = np.zeros(channels)
    for x in tqdm(dataloader):
        x_tot += x[0].mean([0, 2, 3]).cpu().numpy()
        x2_tot += (x[0] ** 2).mean([0, 2, 3]).cpu().numpy()

    channel_avr = x_tot / len(dataloader)
    channel_std = np.sqrt(x2_tot / len(dataloader) - channel_avr**2)
    return channel_avr, channel_std


def main(parameters, args):

    pseudo_wrapper = DefaultWrapper(parameters)
    main_trgt = parameters["dataset_params"]["dataset"]
    for dset, _dset in pseudo_wrapper.dataset_mapper.items():
        if not args.all and dset != main_trgt:
            continue
        print(f"Computing stats for {dset}")

        params = deepcopy(parameters)
        _dset.domain_wise_test = True
        params["dataset_params"]["dataset"] = dset

        data_stats = {}
        for domain in _dset.DNET_DOMAINS:
            print(f"     {domain}")
            params["dataset_params"]["data_test_domain"] = domain
            wrapper = DefaultWrapper(params)
            domset = _dset(wrapper.dataset_params, mode="test")
            _avg, _std = compute_stats(dataset=domset, img_size=args.image_size)
            data_stats[domain] = {"mean": _avg, "std": _std}

        stat_save_path1 = os.path.join(domset.root_dir, "domain_stats.csv")
        stat_save_path2 = os.path.join(domset.root_dir, "domain_stats.pickle")
        save_pickle(data_stats, stat_save_path2)
        data_stats = pd.DataFrame.from_dict(data_stats)
        data_stats.to_csv(stat_save_path1)


if __name__ == "__main__":
    args = parse_arguments()
    parameters = edict(load_json(args.params_path))
    main(parameters, args)
