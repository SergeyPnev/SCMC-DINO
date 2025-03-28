import pdb
import h5py
import random
import skimage
from utils import *
import tifffile as tiff
from .bases import BaseSet
from scipy.io import mmread
from collections import Counter
from datetime import datetime
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor, ToPILImage
import ast

def transform_to_list(val):
    return ast.literal_eval(val)


DATA_INFO = {
    "RxRx1_HUVEC": {"dataset_location": "RXRX1"},
    "CPG0004_full_img_large": {"dataset_location": "CPG0004_full_img_large"},
}


class DomainSet(BaseSet):

    def get_subset(self, dataframe, subset_size=10000, overwrite=False):
        dtvs_dir = os.path.join(
            self.root_dir, "domain_trainval_splits", "subset_splits"
        )
        check_dir(dtvs_dir)
        subset_path = os.path.join(
            dtvs_dir,
            f"{self.name}-test_domain_{self.test_domain}-{self.mode}_subset-{subset_size}.csv",
        )

        if not os.path.isfile(subset_path) or overwrite:
            if is_rank0():
                print('Creating a new subset split for "{}" !'.format(self.name))
                _, subset = train_test_split(dataframe, test_size=subset_size)
                subset.to_csv(subset_path)
            synchronize()
            subset = pd.read_csv(subset_path)

        else:
            subset = pd.read_csv(subset_path)
            if len(subset) != subset_size:
                print_ddp(
                    f'Found updated subset size for "{dataset_name}" for "{dataset_name}"  of size {subset_size} !'
                )
                self.get_subset(dataframe, subset_size=subset_size, overwrite=True)

        return subset

    def get_train_val_splits(
        self, dataframe, val_size=0.1, overwrite=False, allowed_div=1
    ):
        dtvs_dir = os.path.join(self.root_dir, "domain_trainval_splits")
        check_dir(dtvs_dir)
        all_doms = deepcopy(self.DNET_DOMAINS)

        train_data, val_data = [], []
        for didx in range(len(all_doms)):
            domname = all_doms[didx]
            dom_data = dataframe[dataframe.domain.isin([all_doms[didx]])].copy()

            dom_train_path = os.path.join(dtvs_dir, f"{self.name}_{domname}_train.csv")
            dom_val_path = os.path.join(dtvs_dir, f"{self.name}_{domname}_val.csv")

            total_size = len(dom_data)
            int_val_size = int(total_size * val_size)
            int_train_size = int(total_size - int_val_size)

            if (
                not os.path.isfile(dom_train_path)
                or not os.path.isfile(dom_val_path)
                or overwrite
            ):
                if is_rank0():
                    print(
                        'Creating a new train/val split for "{}-{}" !'.format(
                            self.name, domname
                        )
                    )
                    dom_train, dom_val = train_test_split(dom_data, test_size=val_size)
                    dom_train.to_csv(dom_train_path)
                    dom_val.to_csv(dom_val_path)
                synchronize()
                dom_train = pd.read_csv(dom_train_path)
                dom_val = pd.read_csv(dom_val_path)

            else:
                dom_train = pd.read_csv(dom_train_path)
                dom_val = pd.read_csv(dom_val_path)

                train_split_ok = (
                    int_train_size - allowed_div
                    <= len(dom_train)
                    <= int_train_size + allowed_div
                )
                val_split_ok = (
                    int_val_size - allowed_div
                    <= len(dom_val)
                    <= int_val_size + allowed_div
                )
                if not (train_split_ok and val_split_ok):
                    print_ddp(
                        f'Found updated train/validation size for "{dataset_name}" !'
                    )
                    self.get_train_val_splits(
                        dataframe, val_size=val_size, overwrite=True
                    )
            if domname in self.domains:
                train_data.append(dom_train)
                val_data.append(dom_val)

        assert (
            len(train_data) == len(val_data) == self.num_domains - 1
        ), "Mismatch of train/val/test domains"
        train_data = pd.concat(train_data).reset_index(drop=True)
        val_data = pd.concat(val_data).reset_index(drop=True)  #

        return train_data, val_data

    def get_data_as_list(self, use_subset=0):
        data_list = []
        df = self.get_dataframe()

        coresponding_labels_and_names = df[["label", "label_name"]].drop_duplicates()
        self.int_to_labels = dict(
            zip(
                coresponding_labels_and_names.label,
                coresponding_labels_and_names.label_name,
            )
        )
        self.labels_to_int = {val: key for key, val in self.int_to_labels.items()}

        if self.domain_wise_test:
            if self.mode != "test":
                train_data, val_data = self.get_train_val_splits(df, val_size=0.1)
                data = train_data if self.mode == "train" else val_data
            else:
                data = df[df.domain == self.test_domain]
        else:
            data = df

        if use_subset and len(data) > use_subset:
            data = self.get_subset(data, subset_size=use_subset)

        self.df = data
        labels = data["label"].values.tolist()
        img_paths = data["img_path"].values.tolist()
        domain = data["domain"].values.tolist()
        df_index = data.index.values.tolist()

        domains_found, counts_in_each_domain = np.unique(domain, return_counts=True)

        if is_rank0():
            print(domains_found, counts_in_each_domain)

        examples_in_each_domain_dict = {
            self.domain_to_int[dom]: dom_count
            for dom, dom_count in zip(domains_found, counts_in_each_domain)
        }

        examples_in_each_domain = np.zeros(len(self.DNET_DOMAINS))

        for k, v in examples_in_each_domain_dict.items():
            examples_in_each_domain[k] = v

        self.examples_in_each_domain = examples_in_each_domain

        if is_rank0():
            print(self.examples_in_each_domain)

        if self.mode == "train" and not self.fb:
            self.include_info = False
        else:
            self.include_info = False

        self.data_id_2_df_id = {i: val for i, val in enumerate(df_index)}
        self.df_id_2_data_id = {val: key for key, val in self.data_id_2_df_id.items()}

        data_list = [
            {
                "img_path": os.path.join(self.root_dir, str(img_path)),
                "label": label,
                "domain": self.domain_to_int[dom],
                "dataset": self.name,
            }
            for img_path, label, dom in zip(img_paths, labels, domain)
        ]

        return data_list

    def __getitem__(self, idx):
        img_path = self.data[idx]["img_path"]
        cls = torch.as_tensor(self.data[idx]["label"])
        domain = torch.as_tensor(self.data[idx]["domain"])
        label = (cls, domain)
        if self.include_info:
            label = (cls, domain, cls)

        img = self.get_image(img_path)

        if self.cross_batch_training and self.mode == "train" and not self.fb:
            df_idx = self.data_id_2_df_id[idx]
            id_info = self.df.loc[df_idx][["domain", "label"]]

            df_id_sample = int(
                self.df[(self.df.domain != id_info[0]) & (self.df.label == id_info[1])]
                .sample()
                .index.values
            )

            idx_2 = self.df_id_2_data_id[df_id_sample]

            img_path_2 = self.data[idx_2]["img_path"]
            cls_2 = torch.as_tensor(self.data[idx_2]["label"])
            domain_2 = torch.as_tensor(self.data[idx_2]["domain"])
            label = (cls, (domain, domain_2))
            if self.include_info:
                label = (cls, (domain, domain_2), cls)

            img_2 = self.get_image(img_path_2)

        if self.resizing is not None:
            img = self.resizing(img)
            if self.cross_batch_training and self.mode == "train" and not self.fb:
                img_2 = self.resizing(img_2)

        if self.transform is not None:
            if isinstance(self.transform, list):
                if self.cross_batch_training and self.mode == "train" and not self.fb:
                    img_list = [self.transform[0](img), self.transform[1](img_2)]
                    img_list += [tr(img) for tr in self.transform[2:5]]
                    img_list += [tr(img_2) for tr in self.transform[5:]]
                    img = img_list
                else:
                    img = [tr(img) for tr in self.transform]

            else:
                if self.is_multi_crop:
                    img = self.multi_crop_aug(img, self.transform)
                else:
                    img = [self.transform(img) for _ in range(self.num_augmentations)]
            img = img[0] if len(img) == 1 and isinstance(img, list) else img

        return img, label

    def get_image(self, img_path):
        png_path = ".".join(img_path.split(".")[:-1]) + ".png"
        if os.path.exists(png_path):
            img = self.get_x(png_path)
            img_path = png_path
        else:
            img = self.get_x(img_path)
        return img

    def get_dataframe(self):

        if is_rank0():
            print(
                "Test domain: ", self.test_domain, " Training domains: ", self.domains
            )

        combined_list = []
        combined_df_list = []

        for domain in self.DNET_DOMAINS:

            mypath = self.root_dir + "/" + domain + "/"
            classes = os.listdir(mypath)
            for cls in classes:
                cls_path = os.path.join(self.root_dir, domain, cls)
                img_paths = os.listdir(cls_path)
                temp_df = pd.DataFrame(img_paths, columns=["path"])
                temp_df["domain"] = domain
                temp_df["label_name"] = cls
                temp_df["path"] = [
                    os.path.join(domain, cls, p) for p in temp_df["path"]
                ]
                temp_df["img_path"] = temp_df["path"]
                combined_df_list.append(temp_df)

        combined = pd.concat(combined_df_list).reset_index(drop=True)
        unique_ = combined.label_name.unique()
        unique_.sort()
        translate = {x: i for i, x in enumerate(unique_)}
        combined["label"] = combined["label_name"].map(translate)

        return combined

    def get_stats(self):

        stat_save_path = os.path.join(self.root_dir, "domain_stats.pickle")
        stats = load_pickle(stat_save_path)

        mean = []
        std = []
        for dom in self.DNET_DOMAINS:
            if dom not in self.domains and self.mode != "test":
                continue
            if dom != self.test_domain and self.mode == "test":
                continue
            mean.append(stats[dom]["mean"])
            std.append(stats[dom]["std"])

        self.mean = np.mean(mean, axis=0)
        self.std = np.mean(std, axis=0)


class RxRx1_HUVEC(DomainSet):

    name = "RxRx1_HUVEC"
    img_channels = 6
    is_multiclass = True
    cross_batch_training = False
    include_info = True
    task = "classification"
    normalize_per_plate = True
    max_pixel_value = 255.0

    drop_treatment_duplicates = False
    drop_treatment_duplicates_keep_controlls = False

    mean = (0.02051845, 0.07225967, 0.0303462, 0.03936887, 0.01122004, 0.03018961)
    std = (0.02233115, 0.0472001, 0.01045581, 0.02445364, 0.00950606, 0.01063623)

    split_number = 1

    h5_path_x = "/raid/user/data/RXRX1/RXRX1_large_plate_info_corrected_size.hdf5"
    pkl_path = "/raid/user/data/RXRX1/rxrx1/metadata_v2.pkl"
    plate_wise_control_path = (
        "/raid/user/data/RXRX1/plate_negative_control_norm.csv"
    )

#     h5 = True
#     crops_training = False

    h5 = False
    crops_training = True

    domain_wise_test = False
    DNET_DOMAINS = [x for x in range(24)]

    sub_sample = False
    subset_data = False
    subset_strategy = ""

    int_to_labels = {x: int(x) for x in range(1139)}
    int_to_domain = {i: int(x) for i, x in enumerate(DNET_DOMAINS)}
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    domain_to_int = {val: key for key, val in int_to_domain.items()}

    target_metric = "accuracy"
    knn_nhood = 10
    n_classes = len(int_to_labels)

    int_to_id = int_to_labels

    def __init__(self, dataset_params, mode="train", fb=False, use_subset=0):

        self.use_subset = use_subset
        self.attr_from_dict(dataset_params)
        self.dataset_location = DATA_INFO["RxRx1_HUVEC"]["dataset_location"]
        self.root_dir = os.path.join(self.data_location, self.dataset_location)
        if self.crops_training:
            self.root_dir = os.path.join(self.root_dir, "rxrx1/crops")

        self.test_domain = self.data_test_domain
        self.domains = deepcopy(self.DNET_DOMAINS)

        self.num_domains = len(self.domains)

        self.mode = mode
        self.read_mode = "train"
        self.fb = fb
        self.get_stats()
        self.data = self.get_data_as_list(use_subset=self.use_subset)
        self.transform, self.resizing = self.get_transforms(albumentations=True)

        self.h5_file_x = None

        if mode == "test":
            self.return_id = True
        else:
            self.return_id = False

    def get_dataframe(self):

        if self.h5:
            with h5py.File(self.h5_path_x, "r") as full_data:

                h5_file_y = np.array(full_data[self.read_mode + "_labels"])
                h5_file_b = np.array(full_data[self.read_mode + "_batch"])
                h5_file_p = np.array(full_data[self.read_mode + "_plate"])
                length = len(h5_file_y)

                df = pd.DataFrame(
                    {
                        "label_name": h5_file_y[:],
                        "label": h5_file_y[:],
                        "plate": h5_file_p[:],
                        "domain": h5_file_b[:],
                    },
                    columns=["label_name", "label", "plate", "domain"],
                )
            df["img_path"] = df.index

        else:
            df = pd.read_pickle(self.pkl_path)
            df["label_name"] = df["sirna_id"]
            df["label"] = df["sirna_id"]
            df["plate"] = df["Metadata_Plate"]
            df["domain"] = df["experiment"]
            df["crops"] = df["crops"]
            df["img_path"] = df["img_path"]

        all_domains = [i for i in range(24)]

        splits = {
            1: {
                "train": all_domains[0:16],
                "eval": all_domains[16:20],
                "test": all_domains[20:24],
            },
            2: {
                "train": all_domains[4:20],
                "eval": all_domains[20:24],
                "test": all_domains[0:4],
            },
            3: {
                "train": all_domains[8:24],
                "eval": all_domains[0:4],
                "test": all_domains[4:8],
            },
            4: {
                "train": all_domains[12:24] + all_domains[0:4],
                "eval": all_domains[4:8],
                "test": all_domains[8:12],
            },
            5: {
                "train": all_domains[16:24] + all_domains[0:8],
                "eval": all_domains[8:12],
                "test": all_domains[12:16],
            },
            6: {
                "train": all_domains[20:24] + all_domains[0:12],
                "eval": all_domains[12:16],
                "test": all_domains[16:20],
            },
        }

        df = df[df.domain.isin(splits[self.split_number][self.mode])]
#         print(splits[self.split_number][self.mode])

        if self.mode == "train" and not self.fb:
            if self.subset_data:
                if self.subset_strategy == "only_controls":

                    # Only select controls for training
                    df = df[~(df.label_name < 1108)]
                elif self.subset_strategy == "only_treatments":
                    # Only select treatments for training
                    df = df[(df.label_name < 1108)]
                elif self.subset_strategy == "only_treatments_half_as_many":
                    df = df[(df.label_name < 1108)].drop_duplicates(
                        subset=["label", "plate"]
                    )  # Only select treatments for training
                elif (
                    self.subset_strategy == "only_treatments_half_as_many_with_controls"
                ):
                    # Keeps all the controlls and one replicate per plate and
                    # label
                    unint = (
                        df[(df.label_name < 1108)]
                        .drop_duplicates(subset=["label", "plate"])
                        .index
                    )

                    print("NUMBER OF DROPED INDEXES", len(unint))

                    df = df.drop(unint)
                    print("SIZE AFTER DROPPING", df.shape)

                elif (
                    self.subset_strategy == "only_half_as_many_treatments_with_controls"
                ):
                    drop_ids = [
                        i
                        for i in df[(df.label_name < 1108)].label_name.unique()
                        if i % 2 == 0
                    ]
                    df = df[~df.label_name.isin(drop_ids)]

        if self.drop_treatment_duplicates:
            df = df[~(df.label_name < 1108)].drop_duplicates(subset=["label", "plate"])
        if self.drop_treatment_duplicates_keep_controlls:
            # Keeps all the controlls and one replicate per plate and label
            unint = (
                df[(df.label_name < 1108)]
                .drop_duplicates(subset=["label", "plate"])
                .index
            )
            df = df.drop(unint)

        return df

    def get_data_as_list(self, use_subset=0):
        data_list = []
        df = self.get_dataframe()

        coresponding_labels_and_names = df[["label", "label_name"]].drop_duplicates()
        self.int_to_labels = dict(
            zip(
                coresponding_labels_and_names.label,
                coresponding_labels_and_names.label_name,
            )
        )
        self.labels_to_int = {val: key for key, val in self.int_to_labels.items()}

        if self.domain_wise_test:
            if self.mode != "test":
                train_data, val_data = self.get_train_val_splits(df, val_size=0.1)
                data = train_data if self.mode == "train" else val_data
            else:
                data = df[df.domain == self.test_domain]
        else:
            data = df

        if use_subset and len(data) > use_subset:
            data = self.get_subset(data, subset_size=use_subset)

        self.df = data
        labels = data["label"].values.tolist()
        img_paths = data["img_path"].values.tolist()
        domain = data["domain"].values.tolist()
        df_index = data.index.values.tolist()
        if self.crops_training:
            experiments = data["Metadata_Experiment"].values.tolist()
            crops = data["crops"].values.tolist()

        domains_found, counts_in_each_domain = np.unique(domain, return_counts=True)

        if is_rank0():
            print(domains_found, counts_in_each_domain)

        examples_in_each_domain_dict = {
            self.domain_to_int[dom]: dom_count
            for dom, dom_count in zip(domains_found, counts_in_each_domain)
        }

        examples_in_each_domain = np.zeros(len(self.DNET_DOMAINS))

        for k, v in examples_in_each_domain_dict.items():
            examples_in_each_domain[k] = v

        self.examples_in_each_domain = examples_in_each_domain

        if is_rank0():
            print(self.examples_in_each_domain)

        if self.mode == "train" and not self.fb:
            self.include_info = False
        else:
            self.include_info = False

        self.data_id_2_df_id = {i: val for i, val in enumerate(df_index)}
        self.df_id_2_data_id = {val: key for key, val in self.data_id_2_df_id.items()}

        if not self.crops_training:
            data_list = [
                {
                    "img_path": os.path.join(self.root_dir, str(img_path)),
                    "label": label,
                    "domain": self.domain_to_int[dom],
                    "dataset": self.name,
                }
                for img_path, label, dom in zip(img_paths, labels, domain)
            ]

        else:
            data_list = [
                {
                    "img_path": os.path.join(self.root_dir, exp, str(img_path)),
                    "label": label,
                    "domain": self.domain_to_int[dom],
                    "dataset": self.name,
                    "crops": crop
                }
                for img_path, label, dom, crop, exp in zip(img_paths, labels, domain, crops, experiments)
            ]


        return data_list

    def get_stats(self):

        stat_save_path = os.path.join(self.root_dir, "domain_stats.pickle")

        mean = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        std = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self.mean = np.mean(mean, axis=0) * 0.0
        self.std = np.mean(std, axis=0)

        self.norm_df = pd.read_csv(self.plate_wise_control_path)

        self.mean_columns = ["mean_" + str(x) for x in range(6)]
        self.std_columns = ["std_" + str(x) for x in range(6)]

    def get_image(self, img_path):

        if self.h5_file_x is None:
            self.h5_file_x = h5py.File(self.h5_path_x, "r")[self.read_mode + "_img"]
            self.h5_file_plate = h5py.File(self.h5_path_x, "r")[
                self.read_mode + "_plate"
            ]

        img = self.h5_file_x[int(img_path.split("/")[-1])]
        plate = self.h5_file_plate[int(img_path.split("/")[-1])]

        return img, plate

    def plate_normalize_image(self, x, p):
        x = x.float()

        if self.normalize_per_plate:
            plate_stats = self.norm_df[self.norm_df.plate == p]
            mean_vals = np.array([plate_stats[self.mean_columns].values]) * (1 / 255.0)
            std_vals = np.array([plate_stats[self.std_columns].values]) * (1 / 255.0)

        else:
            mean_vals = np.array(
                [
                    [
                        5.23220486,
                        18.42621528,
                        7.73828125,
                        10.0390625,
                        2.86111111,
                        7.69835069,
                    ]
                ]
            ) * (1 / 255.0)
            std_vals = np.array(
                [
                    [
                        5.69444444,
                        12.03602431,
                        2.66623264,
                        6.23567708,
                        2.42404514,
                        2.71223958,
                    ]
                ]
            ) * (1 / 255.0)

        norm_values_mean = (
            torch.tensor(mean_vals).view(-1, 1, 1).type(torch.FloatTensor)
        )
        norm_values_std = (
            torch.tensor(std_vals).view(-1, 1, 1).type(torch.FloatTensor)
        )

        x = (x - norm_values_mean) / (norm_values_std)

        return x

    def get_crops_image(self, img_path, crops_xy, domain):
        image = []
        for xy in crops_xy:
            x, y = xy
            path = img_path + f"_{y}_{x}.tiff"
            image.append(skimage.io.imread(path).astype(np.float32))
        plate = img_path.split("/")[-2][-1]
        plate = domain.numpy() * 10 + int(plate)
        return image, plate

    def __getitem__(self, idx):

        img_path = self.data[idx]["img_path"]
        cls = torch.as_tensor(self.data[idx]["label"])
        domain = torch.as_tensor(self.data[idx]["domain"])
        label = (cls, domain)
        if self.include_info:
            label = (cls, domain, cls)

        if not self.crops_training:
            img, plate = self.get_image(img_path)
        else:
            crops = self.data[idx]["crops"]
            indices = np.random.choice(len(crops), size=64, replace=True)
            crops_xy = [crops[i] for i in indices]
            img, plate = self.get_crops_image(img_path, crops_xy, domain)

        if self.return_id:
            label = (cls, domain, cls)

        if self.cross_batch_training and self.mode == "train" and not self.fb:
            df_idx = self.data_id_2_df_id[idx]
            id_info = self.df.loc[df_idx][["domain", "label"]]

            df_id_sample = int(
                self.df[(self.df.domain != id_info[0]) & (self.df.label == id_info[1])]
                .sample()
                .index.values
            )

            idx_2 = self.df_id_2_data_id[df_id_sample]

            img_path_2 = self.data[idx_2]["img_path"]
            cls_2 = torch.as_tensor(self.data[idx_2]["label"])
            domain_2 = torch.as_tensor(self.data[idx_2]["domain"])
            label = (cls, (domain, domain_2))
            if self.include_info:
                label = (cls, (domain, domain_2), cls)
            if not self.crops_training:
                img_2, plate_2 = self.get_image(img_path_2)
            else:
                crops_2 = self.data[idx_2]["crops"]
                indices_2 = np.random.choice(len(crops_2), size=64, replace=True)
                crops_xy_2 = [crops_2[i] for i in indices_2]
                img_2, plate_2 = self.get_crops_image(img_path_2, crops_xy_2, domain_2)
        elif (not self.cross_batch_training) and self.mode == "train" and not self.fb:
            label = (cls, (domain, domain))

        if self.resizing is not None and self.crops_training:
            crops = [self.resizing(image=single_crop)["image"] for single_crop in img]
            if self.cross_batch_training and self.mode == "train" and not self.fb:
                crops_2 = [self.resizing(image=single_crop)["image"] for single_crop in img_2]

        elif self.resizing is not None and not self.crops_training:
            img = self.resizing(image=img)["image"]
            if self.cross_batch_training and self.mode == "train" and not self.fb:
                img_2 = self.resizing(image=img_2)["image"]

        if self.transform is not None:

            if isinstance(self.transform, list):

                if self.cross_batch_training and self.mode == "train" and not self.fb and not self.crops_training:
                    img_list = [
                        self.plate_normalize_image(
                            (self.transform[0](image=img)["image"]), plate
                        ),
                        self.plate_normalize_image(
                            (self.transform[1](image=img_2)["image"]), plate_2
                        ),
                    ]
                    img_list += [
                        self.plate_normalize_image(tr(image=img)["image"], plate)
                        for tr in self.transform[2:5]
                    ]
                    img_list += [
                        self.plate_normalize_image(tr(image=img_2)["image"], plate_2)
                        for tr in self.transform[5:]
                    ]
                    img = img_list

                elif self.cross_batch_training and self.mode == "train" and not self.fb and self.crops_training:
                    crops = img
                    crops_2 = img_2

                    img_list = []
                    img_list += [torch.stack([self.plate_normalize_image(
                        self.transform[0](image=i)["image"], plate) for i in crops])]
                    img_list += [torch.stack([self.plate_normalize_image(
                        self.transform[1](image=i)["image"], plate_2) for i in crops_2])]

                    for tr in self.transform[2:5]:
                        img_list += [torch.stack([self.plate_normalize_image(
                        tr(image=i)["image"], plate) for i in crops])]

                    for tr in self.transform[5:8]:
                        img_list += [torch.stack([self.plate_normalize_image(
                        tr(image=i)["image"], plate_2) for i in crops_2])]
                    img = img_list

                else:
                    img = [
                        self.plate_normalize_image(tr(image=img)["image"], plate)
                        for tr in self.transform
                    ]

            else:
                if self.is_multi_crop:
                    img = self.multi_crop_aug(img, self.transform)
                elif self.crops_training:
                    # 96 crops get image
                    img = torch.stack([self.plate_normalize_image(
                        self.transform(image=i)["image"], plate) for i in crops])
                else:
                    img = [
                        self.plate_normalize_image(
                            self.transform(image=img)["image"], plate
                        )
                        for _ in range(self.num_augmentations)
                    ]

        if self.cross_batch_training and self.mode == "train" and not self.fb:
            img = [im.float() for im in img]
            img = img[0] if len(img) == 1 and isinstance(img, list) else img
        else:
            img = img.float()
            
        if img[0].dtype != torch.float32 or img[1].dtype != torch.float32:
            print(img[0].dtype)

        return img, label


class CPG0004_full_img_large(BaseSet):
    img_channels = 5
    is_multiclass = True
    include_info = False
    return_id = True
    task = "classification"

    cross_batch_training = True
    phase_training = False
    triplets_training = False

    crops_training = False
#     crops_training = False
    dmso_training = False
#     local_crops = True

    n_cells = 8

    mean = (14.61494155, 28.87414807, 32.61800756, 36.41028929, 22.27769592)
    std = (28.80363038, 31.39763568, 32.06969697, 32.35857192, 25.8217434)
    max_pixel_value = 255.0
    int_to_labels = {i: str(i + 1) for i in range(570)}
    num_domains = 136
    normalize_per_plate = True
    subset_test = False
    predict_moa = False
    cross_validation = True
    split_number = 2
    ratios = (0.2, 0.3)

    if predict_moa:
        int_to_labels = {i: str(i + 1) for i in range(50)}

    target_metric = "accuracy"
    knn_nhood = 10
    n_classes = 571
    labels_to_int = {val: key for key, val in int_to_labels.items()}
    int_to_id = int_to_labels
    domain_to_labels = {i: str(i + 1) for i in range(num_domains)}

    def __init__(self, dataset_params, fb=False, fb_center=False, df_dmso=False, mode="train", use_subset=0):
        self.attr_from_dict(dataset_params)
        self.dataset_location = DATA_INFO["CPG0004_full_img_large"]["dataset_location"]

        # 96 crops get image
        self.root_dir = "/raid/user/data/JUMP/cpg0004-lincs/inputs_224/"
        self.fb = fb
        self.img_size = 224
        self.mode = mode
        self.fb_center = fb_center
        self.get_stats()

        path = "/raid/user/data/JUMP/cpg0004-lincs/metadata/df_withDMSO_withCrops_filtered_v3.csv"

        if not df_dmso:
            self.data = self.get_data_as_list(path, mode=mode)
        else:
            self.data = self.get_data_as_list(path, dmso_flag=True, mode=mode)
#         self.dmso_data = self.get_data_as_list(path, dmso_flag=True, mode="train")

        if self.dict_control and self.mode not in ["eval", "test"] and not fb:
            self.treatment_to_domains = self.get_data_as_dict(self.df, mode="train")

        self.transform, self.resizing = self.get_transforms(albumentations=True)

        if mode == "train" and not fb:
            self.return_id = False
        else:
            self.return_id = True

    def prepare_df_dict(self, df, dmso=False):
        df_dict = {}
        for domain in df["domain"].unique():
            domain_dict = {}
            if not dmso:
                for treatment in df["label"].unique():
                    domain_treatment_df = df[(df["domain"] != int(domain)) & (df["label"] == int(treatment))]
                    try:
                        x = domain_treatment_df.sample().index.values
                    except:
                        domain_treatment_df = df[df["label"] == int(treatment)]
                    domain_dict[treatment] = domain_treatment_df
            else:
                domain_treatment_df = df[df["domain"] == int(domain)]
                domain_dict = domain_treatment_df
            df_dict[domain] = domain_dict
        return df_dict

    def get_data_as_dict(self, df, mode="train"):
        treatment_to_domains = {}
        for treatment, group in df.groupby('label'):
            domain_groups = group.groupby('domain')
            treatment_to_domains[treatment] = {}
            for domain, sub_group in domain_groups:
                treatment_to_domains[treatment][domain] = sub_group.index.tolist()
        return treatment_to_domains

    def get_data_as_list(self, path, dmso_flag=False, mode="train"):
        data_list = []
        datainfo = pd.read_csv(path, index_col=0, engine="python")

        if mode == "eval" or mode == "test":
            datainfo = datainfo[datainfo["Compound"] != "DMSO"].reset_index(drop=True)

        if mode == "train":
            if dmso_flag:
                datainfo = datainfo[datainfo["Compound"] == "DMSO"].reset_index(drop=True)
            else:
                datainfo = datainfo[datainfo["Compound"] != "DMSO"].reset_index(drop=True)

        if self.crops_training:
            datainfo["crops"] = datainfo["crops"].apply(transform_to_list)

        domainlist = datainfo.Metadata_Plate_ID.values
        moa_list = datainfo.Metadata_moa_ID.values
        labellist = datainfo.Treatment_ID.tolist()
        img_names = datainfo.combined_paths.tolist()

        if self.crops_training:
            crops = datainfo.crops.tolist()

        if self.predict_moa:
            labellist = datainfo.Metadata_moa_ID.values.tolist()

        if self.subset_test:
            moa_list = datainfo.index.values

        if self.cross_validation:
            split = datainfo.replicate_ID.tolist()

            if self.crops_training:
                dataframe = pd.DataFrame(
                    list(zip(img_names, labellist, domainlist, moa_list, split, crops)),
                    columns=["img_path", "label", "domains", "moa", "split", "crops"],
                )
            else:
                dataframe = pd.DataFrame(
                    list(zip(img_names, labellist, domainlist, moa_list, split)),
                    columns=["img_path", "label", "domains", "moa", "split"],
                )

            splits = {
                1: {"train": [1.0, 2.0, 3.0], "eval": [4.0], "test": [5.0]},
                2: {"train": [2.0, 3.0, 4.0], "eval": [5.0], "test": [1.0]},
                3: {"train": [3.0, 4.0, 5.0], "eval": [1.0], "test": [2.0]},
                4: {"train": [1.0, 4.0, 5.0], "eval": [2.0], "test": [3.0]},
                5: {"train": [1.0, 2.0, 5.0], "eval": [3.0], "test": [4.0]},
           }

            train_ids = dataframe[
                dataframe.split.isin(splits[self.split_number]["train"])
            ].index.values
            val_ids = dataframe[
                dataframe.split.isin(splits[self.split_number]["eval"])
            ].index.values
            test_ids = dataframe[
                dataframe.split.isin(splits[self.split_number]["test"])
            ].index.values

            if self.mode == "train":
                data = dataframe.loc[train_ids, :]
            elif self.mode in ["val", "eval"]:
                data = dataframe.loc[val_ids, :]
            else:
                data = dataframe.loc[test_ids, :]

        else:
            split = datainfo.Split.tolist()
            if self.crops_training:
                dataframe = pd.DataFrame(
                    list(zip(img_names, labellist, domainlist, moa_list, split, crops)),
                    columns=["img_path", "label", "domains", "moa", "split", "crops"],
                )
            else:
                dataframe = pd.DataFrame(
                    list(zip(img_names, labellist, domainlist, moa_list, split)),
                    columns=["img_path", "label", "domains", "moa", "split"],
                )

            train_ids = dataframe[dataframe.split == "Training"].index.values
            val_ids = dataframe[dataframe.split == "Validation"].index.values
            test_ids = dataframe[dataframe.split == "Test"].index.values

            if self.mode == "train":
                data = dataframe.loc[train_ids, :]
            elif self.mode in ["val", "eval"]:
                data = dataframe.loc[val_ids, :]
            else:
                data = dataframe.loc[test_ids, :]

        labels = data["label"].values.tolist()
        domains = data["domains"].values.tolist()
        img_paths = data["img_path"].values.tolist()
        moa_data = data["moa"].values.tolist()
        split_data = data["split"].values.tolist()
        if self.crops_training:
            crops = data["crops"].values.tolist()

        domains_found, counts_in_each_domain = np.unique(domains, return_counts=True)

        examples_in_each_domain_dict = {
            dom: dom_count
            for dom, dom_count in zip(domains_found, counts_in_each_domain)
        }

        examples_in_each_domain = np.zeros(
            self.num_domains
        )

        for k, v in examples_in_each_domain_dict.items():
            examples_in_each_domain[k] = v

        if not dmso_flag:
            self.examples_in_each_domain = examples_in_each_domain

        if self.crops_training:
            data_list = [
                {
                    "img_path": img_path,
                    "label": label,
                    "domain": domain,
                    "dataset": self.name,
                    "moa": moa,
                    "split": split,
                    "crops": crop,
                }
                for img_path, label, domain, moa, split, crop in zip(
                    img_paths, labels, domains, moa_data, split_data, crops
                )
        ]
        else:
            data_list = [
                    {
                        "img_path": img_path,
                        "label": label,
                        "domain": domain,
                        "dataset": self.name,
                        "moa": moa,
                        "split": split,
                    }
                    for img_path, label, domain, moa, split in zip(
                        img_paths, labels, domains, moa_data, split_data
                    )
            ]

        self.df = pd.DataFrame(data_list)

        return data_list

    def get_stats(self):
        path = "/raid/user/data/JUMP/cpg0004-lincs/metadata/top20_moa_plate_norms_dmso.csv"
        datainfo = pd.read_csv(path, index_col=0, engine="python")

        mean = [1.0, 1.0, 1.0, 1.0, 1.0]
        std = [1.0, 1.0, 1.0, 1.0, 1.0]

        self.mean = np.mean(mean, axis=0) * 0.0
        self.std = np.mean(std, axis=0)

        self.norm_df = datainfo
        self.norm_df["plate"] = [i for i in range(len(self.norm_df))]

        if self.img_channels == 5:
            self.mean_columns = ["DNA_mean", "ER_mean", "RNA_mean", "AGP_mean", "Mito_mean"]
            self.std_columns = ["DNA_std", "ER_std", "RNA_std", "AGP_std", "Mito_std"]
        elif self.img_channels == 10:
            self.mean_columns = ["DNA_mean", "ER_mean", "RNA_mean", "AGP_mean", "Mito_mean", "DNA_phase_mean", "ER_phase_mean", "RNA_phase_mean", "AGP_phase_mean", "Mito_phase_mean"]
            self.std_columns = ["DNA_std", "ER_std", "RNA_std", "AGP_std", "Mito_std", "DNA_phase_std", "ER_phase_std", "RNA_phase_std", "AGP_phase_std", "Mito_phase_std"]

    def plate_normalize_image(self, x, p):
        x = x.float()
        if self.normalize_per_plate:

            plate_stats = self.norm_df[self.norm_df.plate == p]

            mean_vals = (plate_stats[self.mean_columns].values) * (1 / 255.0)
            std_vals = (plate_stats[self.std_columns].values) * (1 / 255.0)

            norm_values_mean = (
                torch.tensor(np.array([mean_vals]))
                .view(5, 1, 1)
                .type(torch.FloatTensor)
            )
            norm_values_std = (
                torch.tensor(np.array([std_vals])).view(5, 1, 1).type(torch.FloatTensor)
            )

            x = (x - norm_values_mean) / (norm_values_std)
        else:

            mean_vals = np.array(
                [14.61494155, 28.87414807, 32.61800756, 36.41028929, 22.27769592]
            ) * (1 / 255.0)
            std_vals = np.array(
                [28.80363038, 31.39763568, 32.06969697, 32.35857192, 25.8217434]
            ) * (1 / 255.0)

            norm_values_mean = (
                torch.tensor(np.array([mean_vals]))
                .view(5, 1, 1)
                .type(torch.FloatTensor)
            )
            norm_values_std = (
                torch.tensor(np.array([std_vals])).view(5, 1, 1).type(torch.FloatTensor)
            )

            x = (x - norm_values_mean) / (norm_values_std)

        return x

    def check_missing(self, df):
        def check_file(root, img_paths):
            for i, path in enumerate(img_paths.split(",")):
                flag = os.path.isfile(self.root_dir + path)
                if not flag:
                    return img_paths.split(",")[0].split("/")
                else:
                    return 0, 0

        missing, indexes = [], []
        for idx, row in df.iterrows():
            plate, img = check_file(self.root_dir, row["combined_paths"])
            if plate != 0:
                missing.append((plate, img))
                indexes.append(idx)

        new_df = df.copy()
        for idx in indexes:
            new_df = new_df.drop(idx)
        new_df = new_df.reset_index(drop=True)
        return new_df

    def get_image(self, root_dir, img_paths):
        full = np.zeros((self.img_size, self.img_size, 5))
        for i, path in enumerate(img_paths.split(",")):
            full[:, :, i] = skimage.io.imread(root_dir + path)
        return full

    def sample_dmso_idx(self, df, domain, same_idx, just_random=False):
        try:
            if not just_random:
                if (
                    df[(df.domain == int(domain))]
                ).shape[0] > 0:
                    df_id_sample = int(
                        df[
                            (df.domain == int(domain))
                        ]
                        .sample()
                        .index.values
                    )
                else:
                    df_id_sample = same_idx
            else:
                df_id_sample = int(
                    df[
                        (df.domain != int(domain))
                    ]
                    .sample()
                    .index.values
                )
        except:
            df_id_sample = same_idx
        return df_id_sample

    def crop_single_cell(self, image, x, y, scale=0.21, crop_size=10):
        half_crop = crop_size // 2
        x *= scale
        y *= scale
        x = int(x)
        y = int(y)
        img_height, img_width, img_depth = image.shape

        x_min = max(0, x - half_crop)
        x_max = min(img_height, x + half_crop)
        y_min = max(0, y - half_crop)
        y_max = min(img_width, y + half_crop)

        cropped_img = image[x_min:x_max, y_min:y_max, :]

        if cropped_img.shape[0] != crop_size or cropped_img.shape[1] != crop_size:
            cropped_img = np.pad(cropped_img,
                                 ((0, crop_size - cropped_img.shape[0]),
                                  (0, crop_size - cropped_img.shape[1]),
                                  (0, 0)),
                                 mode='constant')
        return cropped_img

    def get_crops_image(self, root_dir, img_paths, crops_xy):
        image = []
        for xy in crops_xy:
            x, y = xy
            path = img_paths.split(",")[0]
            new_path = f"{path[:-4]}_{x}_{y}.tiff"
            image.append(skimage.io.imread(root_dir + new_path).astype(np.float32))
        return image

    def get_crops_image_v2(self, root_dir, idxs):
        image = []
        for i in idxs:
            row = self.data[i]
            img_path = row['img_path']
            x, y = random.choice(row['crops'])
            path = img_path.split(",")[0]
            new_path = f"{path[:-4]}_{x}_{y}.tiff"
            image.append(skimage.io.imread(root_dir + new_path).astype(np.float32))
        return image

    def __getitem__(self, idx):
        img_path = self.data[idx]["img_path"]
        label = torch.as_tensor(self.data[idx]["label"])
        domain = torch.as_tensor(self.data[idx]["domain"])
        moa = torch.as_tensor(self.data[idx]["moa"])
        treatment = self.data[idx]["label"]
        plate = self.data[idx]["domain"]
        split = self.data[idx]["split"]
        if (self.crops_training and not self.dict_control) or (self.mode in ["eval", "test"] and self.crops_training) or (self.fb and self.crops_training):
            crops = self.data[idx]["crops"]
            if self.mode == "train" and not self.fb:
                indices = np.random.choice(len(crops), size=self.n_cells, replace=False)
                crops_xy = [crops[i] for i in indices]
            if self.mode in ["eval", "test"] or self.fb:
                indices = np.random.choice(len(crops), size=8, replace=True)
                crops_xy = [crops[i] for i in indices]
            img = self.get_crops_image(self.root_dir, img_path, crops_xy)
        elif self.crops_training and self.dict_control and self.mode == "train":
            domains = self.treatment_to_domains[treatment]
            available_domains = list(domains.keys())
            remaining_domains = [d for d in available_domains if d != plate]
            idxs = np.random.choice(domains[plate], size=8, replace=True)
            img = self.get_crops_image_v2(self.root_dir, idxs)
        else:
            img = self.get_image(self.root_dir, img_path)

        if self.phase_training:
            img_phase = self.get_image(self.root_dir_phase, img_path)
            img_phase = img_phase.astype(np.float32)

#         label = (label, domain, moa, split, crops_xy)
        label = (label, domain, moa, split)

        # also for train
        if self.cross_batch_training and self.mode == "train" and not self.fb:
            if not self.dict_control:
                try:
                    df_sampling = self.df[(self.df.domain != int(domain)) & (self.df.label == int(label[0]))]
                    if df_sampling.shape[0] > 0:
                        df_id_sample = int(df_sampling.sample().index.values)
                    else:
                        df_id_sample = int(
                            self.df[(self.df.label == int(label[0]))].sample().index.values
                        )
                except:
                    df_id_sample = int(
                        self.df[(self.df.label == int(label[0]))].sample().index.values
                    )

                idx_2 = df_id_sample
                img_path_2 = self.data[idx_2]["img_path"]
                cls_2 = torch.as_tensor(self.data[idx_2]["label"])
                plate_2 = self.data[idx_2]["domain"]
                domain_2 = torch.as_tensor(self.data[idx_2]["domain"])

                if self.meta_learning:
                    try:
                        df_sampling = self.df[(self.df.domain != int(domain)) & (self.df.domain != int(domain_2)) &(self.df.label == int(label[0]))]
                        if df_sampling.shape[0] > 0:
                            df_id_sample = int(df_sampling.sample().index.values)
                        else:
                            df_id_sample = int(
                                self.df[(self.df.label == int(label[0]))].sample().index.values
                            )
                    except:
                        df_id_sample = int(
                            self.df[(self.df.label == int(label[0]))].sample().index.values
                        )
                    idx_3 = df_id_sample
                    img_path_3 = self.data[idx_3]["img_path"]
                    cls_3 = torch.as_tensor(self.data[idx_3]["label"])
                    plate_3 = self.data[idx_3]["domain"]
                    domain_3 = torch.as_tensor(self.data[idx_3]["domain"])

                if self.crops_training:
                    crops_2 = self.data[idx_2]["crops"]
                    indices_2 = np.random.choice(len(crops_2), size=self.n_cells, replace=False)
                    crops_xy_2 = [crops_2[i] for i in indices_2]
                    img_2 = self.get_crops_image(self.root_dir, img_path_2, crops_xy_2)
                else:
                    img_2 = self.get_image(self.root_dir, img_path_2)
                if self.meta_learning:
                    crops_3 = self.data[idx_3]["crops"]
                    indices_3 = np.random.choice(len(crops_3), size=self.n_cells, replace=False)
                    crops_xy_3 = [crops_3[i] for i in indices_3]
                    img_3 = self.get_crops_image(self.root_dir, img_path_3, crops_xy_3)

            elif self.dict_control:
                idxs_2 = []
                if len(remaining_domains) == 2:
                    idxs_2 = random.choice(domains[remaining_domains])
                elif len(remaining_domains) == 1:
                    idxs_2 = domains[remaining_domains[0]]
                    idxs_2 = np.random.choice(idxs_2, size=8, replace=True)
                img_2 = self.get_crops_image_v2(self.root_dir, idxs_2)
                
            if self.dmso_training and self.mode == "train":
                idx_dmso_1 = self.sample_dmso_idx(self.df_dmso, domain, idx, just_random=False)
                idx_dmso_2 = self.sample_dmso_idx(self.df_dmso, domain_2, df_id_sample, just_random=False)
                img_path_dmso_1 = self.dmso_data[idx_dmso_1]["img_path"]
                img_path_dmso_2 = self.dmso_data[idx_dmso_2]["img_path"]

                if self.crops_training:
                    crops_dmso_xy_1 = random.sample(self.dmso_data[idx_dmso_1]["crops"], 8)
                    crops_dmso_xy_2 = random.sample(self.dmso_data[idx_dmso_2]["crops"], 8)
                    crops_dmso_1 = self.get_crops_image(self.root_dir, img_path_dmso_1, crops_dmso_xy_1)
                    crops_dmso_2 = self.get_crops_image(self.root_dir, img_path_dmso_2, crops_dmso_xy_2)

            if self.phase_training:
                img_phase_2 = self.get_image(self.root_dir_phase, img_path_2)
                img_phase_2 = img_phase_2.astype(np.float32)

            if self.meta_learning:
                label = (label, (domain, domain_2, domain_3))
            else:
                label = (label, (domain, domain_2))

        elif (not self.cross_batch_training) and self.mode == "train" and not self.fb:
            label = (label, (domain, domain, domain))

        if self.resizing is not None and self.crops_training:
            crops = [self.resizing(image=single_crop)["image"] for single_crop in img]

            if self.cross_batch_training and self.mode == "train" and not self.fb:
                crops_2 = [self.resizing(image=single_crop)["image"] for single_crop in img_2]
                if self.meta_learning:
                    crops_3 = [self.resizing(image=single_crop)["image"] for single_crop in img_3]

                if self.dmso_training:
                    crops_dmso_1 = [self.resizing(image=crops_dmso_1["image"]) for x,y in crops_dmso_xy_1]
                    crops_dmso_2 = [self.resizing(image=crops_dmso_2["image"]) for x,y in crops_dmso_xy_2]

        if self.resizing is not None and not self.crops_training:
            img = self.resizing(image=img)["image"]
            if self.cross_batch_training and self.mode == "train" and not self.fb:
                img_2 = self.resizing(image=img_2)["image"]
                if self.dmso_training:
                    img_dmso_1 = self.resizing(image=crops_dmso_1)["image"]
                    img_dmso_2 = self.resizing(image=crops_dmso_2)["image"]

        if self.transform is not None:
            if isinstance(self.transform, list):
                if self.cross_batch_training and self.mode == "train" and not self.fb and not self.crops_training:
                    img_list = [
                        self.plate_normalize_image(
                            (self.transform[0](image=img)["image"]), plate
                        ),
                        self.plate_normalize_image(
                            (self.transform[1](image=img_2)["image"]), plate_2
                        ),
                    ]
                    img_list += [
                        self.plate_normalize_image(tr(image=img)["image"], plate)
                        for tr in self.transform[2:5]
                    ]
                    img_list += [
                        self.plate_normalize_image(tr(image=img_2)["image"], plate_2)
                        for tr in self.transform[5:]
                    ]
                    img = img_list

                    if self.dmso_training:
                        img_dmso = [
                            self.plate_normalize_image(
                                (self.transform[0](image=img_dmso_1)["image"]), plate
                            ),
                            self.plate_normalize_image(
                                (self.transform[1](image=img_dmso_1)["image"]), plate_2
                            ),
                        ]

                elif self.cross_batch_training and self.mode == "train" and not self.fb and self.crops_training and not self.meta_learning:
                    crops = img
                    crops_2 = img_2

                    img_list = []
                    img_list += [torch.stack([self.plate_normalize_image(
                         self.transform[0](image=i)["image"], plate) for i in crops])]
                    img_list += [torch.stack([self.plate_normalize_image(
                         self.transform[1](image=i)["image"], plate_2) for i in crops_2])]

                    if self.local_crops:
                        for tr in self.transform[2:5]:
                            img_list += [torch.stack([self.plate_normalize_image(
                                 tr(image=i)["image"], plate) for i in crops])]

                        for tr in self.transform[5:8]:
                            img_list += [torch.stack([self.plate_normalize_image(
                                 tr(image=i)["image"], plate_2) for i in crops_2])]
                    img = img_list

                    if self.dmso_training:
                        img_dmso = []
                        img_dmso += [torch.stack([self.plate_normalize_image(
                            self.transform[0](image=i)["image"], plate) for i in crops_dmso_1])]
                        img_dmso += [torch.stack([self.plate_normalize_image(
                            self.transform[1](image=i)["image"], plate_2) for i in crops_dmso_2])]

                        for tr in self.transform[2:5]:
                            img_dmso += [torch.stack([self.plate_normalize_image(
                                 tr(image=i)["image"], plate) for i in crops_dmso_1])]

                        for tr in self.transform[5:8]:
                            img_dmso += [torch.stack([self.plate_normalize_image(
                                 tr(image=i)["image"], plate_2) for i in crops_dmso_2])]

                elif self.cross_batch_training and self.mode == "train" and not self.fb and self.crops_training and self.meta_learning:
                    crops = img
                    crops_2 = img_2
                    crops_3 = img_3

                    img_list = []
                    img_list += [torch.stack([self.plate_normalize_image(
                         self.transform[0](image=i)["image"], plate) for i in crops])]
                    img_list += [torch.stack([self.plate_normalize_image(
                         self.transform[1](image=i)["image"], plate_2) for i in crops_2])]
                    img_list += [torch.stack([self.plate_normalize_image(
                         self.transform[2](image=i)["image"], plate_3) for i in crops_3])]

                    if self.local_crops:
                        for tr in self.transform[3:6]:
                            img_list += [torch.stack([self.plate_normalize_image(
                                 tr(image=i)["image"], plate) for i in crops])]

                        for tr in self.transform[6:9]:
                            img_list += [torch.stack([self.plate_normalize_image(
                                 tr(image=i)["image"], plate_2) for i in crops_2])]

                        for tr in self.transform[9:12]:
                            img_list += [torch.stack([self.plate_normalize_image(
                                tr(image=i)["image"], plate_3) for i in crops_3])]

                    img = img_list

                else:
                    img = self.plate_normalize_image(self.transform[0](image=img)["image"], plate)

            else:
                if self.is_multi_crop:
                    img = self.multi_crop_aug(img, self.transform)
                elif self.crops_training:
                    img = torch.stack([self.plate_normalize_image(
                        self.transform(image=i)["image"], plate) for i in crops])
                elif len(img) == 1:
                    img = self.plate_normalize_image(self.transform(image=img)["image"], plate)
                else:
                    img = [self.plate_normalize_image(self.transform(image=img)["image"], plate)
                        for _ in range(self.num_augmentations)]

            if self.cross_batch_training and self.mode == "train" and not self.fb:
                img = [im.float() for im in img]
                img = img[0] if len(img) == 1 and isinstance(img, list) else img
            else:
                img = img.float()

        if self.dmso_training and self.mode == "train" and not self.fb:
            img_dmso = [im.float() for im in img_dmso]
            return img, label, img_dmso
        else:
            return img, label

    def __len__(self):
        return self.df.shape[0]
