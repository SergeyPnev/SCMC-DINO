v#!/usr/bin/env python
# coding: utf-8

import argparse
from defaults import *
from helpfuns import *
import sys

sys.path.append("..")

ACCEPED_IMG_TYPES = (".png", ".jpg")


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


class Resizer(BaseSet):

    def __init__(
        self,
        parameters,
        image_size=(512, 512),
        interpolation_method=InterpolationMode.LANCZOS,
        resize_all=False,
    ):

        self.resize_all = resize_all
        self.parameters = edict(parameters)

        self.get_data_as_list()

        self.image_size = image_size
        self.interpolation_method = interpolation_method
        self.resize = Resize(image_size, interpolation=interpolation_method)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        dataset = self.data[idx]["dataset"]
        img_path = self.data[idx]["img_path"]

        png_path = img_path
        if not img_path.endswith(".png"):
            png_path = ".".join(img_path.split(".")[:-1]) + ".png"

        try:
            img = self.datasets[dataset].get_x(img_path)
        except BaseException:
            img = self.datasets[dataset].get_x(png_path)

        if (img.size != self.image_size) or (not img_path.endswith(".png")):
            if img.size != self.image_size:
                img = self.resize(img)
            if not os.path.exists(png_path):
                img.save(png_path)
            # if os.path.exists(img_path) and img_path != png_path:
            #    os.remove(img_path)

        return 0

    def get_data_as_list(self):
        self.data = []
        self.datasets = {}
        wrapper = DefaultWrapper(self.parameters)
        main_trgt = self.parameters["dataset_params"]["dataset"]
        for dset, _dset in wrapper.dataset_mapper.items():
            if not self.resize_all and dset != main_trgt:
                continue
            print(f"Adding {dset} data")
            params = deepcopy(self.parameters)
            params["dataset_params"]["dataset"] = dset
            params["dataset_params"]["data_test_domain"] = _dset.DNET_DOMAINS[0]
            wrapper = DefaultWrapper(params)
            fullset = _dset(wrapper.dataset_params)
            all_subtrees = glob(
                os.path.join(fullset.root_dir, "**", "*"), recursive=True
            )
            self.datasets[dset] = fullset
            all_img_files = [
                {"img_path": f, "dataset": dset}
                for f in all_subtrees
                if f.endswith(ACCEPED_IMG_TYPES)
            ]
            self.data += all_img_files


def main(parameters, args):

    resizer = Resizer(
        parameters, image_size=(args.image_size, args.image_size), resize_all=args.all
    )

    dummloader = DataLoader(
        resizer, shuffle=False, num_workers=args.num_workers, batch_size=1
    )
    for d in tqdm(dummloader, desc="Reshaping and saving data", total=len(dummloader)):
        pass


if __name__ == "__main__":
    args = parse_arguments()
    parameters = edict(load_json(args.params_path))
    main(parameters, args)
