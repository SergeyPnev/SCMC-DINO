import cv2
import umap
import umap.plot
import wandb
import math
import pandas as pd
from .helpfuns import *
from .dist_utills import *
import random
from PIL import ImageOps, ImageFilter
from torch import nn
import plotly.express as px
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.transforms import RandomApply, RandomResizedCrop
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


def compute_stats(dataloader):
    from tqdm import tqdm

    channels = dataloader.dataset[0][0].size(0)
    x_tot = np.zeros(channels)
    x2_tot = np.zeros(channels)
    for x in tqdm(dataloader):
        x_tot += x[0].mean([0, 2, 3]).cpu().numpy()
        x2_tot += (x[0] ** 2).mean([0, 2, 3]).cpu().numpy()

    channel_avr = x_tot / len(dataloader)
    channel_std = np.sqrt(x2_tot / len(dataloader) - channel_avr**2)
    return channel_avr, channel_std


def pil_loader(img_path, n_channels):
    with open(img_path, "rb") as f:
        img = Image.open(f)
        if n_channels == 3:
            return img.convert("RGB")
        elif n_channels == 1:
            return img.convert("L")
        elif n_channels == 4:
            return img.convert("RGBA")
        else:
            raise NotImplementedError(
                "PIL only supports 1,3 and 4 channel inputs. Use cv2 instead"
            )


def model_to_CPU_state(net):
    """Gets the state_dict to CPU for easier save/load later."""
    if is_parallel(net):
        state_dict = {k: deepcopy(v.cpu()) for k, v in net.module.state_dict().items()}
    else:
        state_dict = {k: deepcopy(v.cpu()) for k, v in net.state_dict().items()}
    return OrderedDict(state_dict)


def opimizer_to_CPU_state(opt):
    """Gets the state_dict to CPU for easier save/load later."""
    state_dict = {}
    state_dict["state"] = {}
    state_dict["param_groups"] = deepcopy(opt.state_dict()["param_groups"])

    for k, v in opt.state_dict()["state"].items():
        state_dict["state"][k] = {}
        if v:
            for _k, _v in v.items():
                if torch.is_tensor(_v):
                    elem = deepcopy(_v.cpu())
                else:
                    elem = deepcopy(_v)
                state_dict["state"][k][_k] = elem
    return state_dict


def load_state_dict(model, state_dict, strict=True):
    if is_parallel(model):
        dif_keys = model.module.load_state_dict(state_dict, strict=strict)
    else:
        dif_keys = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        return dif_keys


class MovingMeans:
    def __init__(self, window=5):
        self.window = window
        self.values = []

    def add(self, val):
        self.values.append(val)

    def get_value(self):
        return np.convolve(
            np.array(self.values), np.ones((self.window,)) / self.window, mode="valid"
        )[-1]


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def __call__(self, running_module, ma_module):
        if ma_module is None:
            return new
        return ma_module * self.beta + (1 - self.beta) * running_module


def compute_stats(dataloader):
    from tqdm import tqdm

    channels = dataloader.dataset[0]["img"].size(0)
    x_tot = np.zeros(channels)
    x2_tot = np.zeros(channels)
    for x in tqdm(dataloader):
        x_tot += x["img"].mean([0, 2, 3]).cpu().numpy()
        x2_tot += (x["img"] ** 2).mean([0, 2, 3]).cpu().numpy()

    channel_avr = x_tot / len(dataloader)
    channel_std = np.sqrt(x2_tot / len(dataloader) - channel_avr**2)
    return channel_avr, channel_std


def conv2d_kaiming_uniform_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")


def conv2d_kaiming_normal_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")


class LinearWarmup(_LRScheduler):
    __name__ = "LinearWarmup"

    def __init__(
        self,
        optimizer,
        max_lr,
        warmup_iters=0,
        warmup_epochs=0,
        eta_min=1e-8,
        last_epoch=-1,
        verbose=False,
        steps_per_epoch=None,
    ):
        if warmup_iters and warmup_epochs:
            print_ddp(
                "\033[93m Found nonzero arguments for warmup_iters and warmup_epochs \033[0m"
            )
            print_ddp("\033[93m Using warmup_epochs instead of warmup_iters \033[0m")
            warmup_iters = steps_per_epoch * warmup_epochs
        if not warmup_iters and not warmup_epochs:
            print_ddp(
                "\033[93m No warmup period found but LinearWarmup is used \033[0m"
            )
            warmup_iters = 1
        else:
            if warmup_epochs and steps_per_epoch is None:
                raise TypeError(
                    "LinearWarmup with warmup_epochs settings must include steps_per_epoch"
                )
            elif warmup_epochs and steps_per_epoch is not None:
                warmup_iters = steps_per_epoch * warmup_epochs

        self.warmup_iters = warmup_iters
        self.eta_min = eta_min
        self.max_lr = max_lr
        for group in optimizer.param_groups:
            group["lr"] = self.eta_min
        super(LinearWarmup, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == -1:
            return [self.eta_min for group in self.optimizer.param_groups]
        elif self.last_epoch > self.warmup_iters:
            return [group["lr"] for group in self.optimizer.param_groups]
        else:
            return [
                group["lr"] + (1 / self.warmup_iters) * (self.max_lr - self.eta_min)
                for group in self.optimizer.param_groups
            ]

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == -1:
            return [self.eta_min for group in self.optimizer.param_groups]
        elif self.last_epoch > self.warmup_iters:
            return [group["lr"] for group in self.optimizer.param_groups]
        else:
            return [
                group["lr"] + (1 / self.warmup_iters) * (self.max_lr - self.eta_min)
                for group in self.optimizer.param_groups
            ]


def rand_apply(tranform, p=0.5):
    return RandomApply(torch.nn.ModuleList([tranform]), p)


class _RandomSolarize:
    # moving it to _ since torchvision has RandomSolarize in the new releases
    # (>1.8)
    def __init__(self, threshold=128, p=0.5):
        self.threshold = threshold
        self.p = p

    def __call__(self, sample):
        if self.p < random.random():
            return sample
        return ImageOps.solarize(sample, self.threshold)


class RandomGaussianBlur:
    # copy-past from https://github.com/facebookresearch/dino
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def shift_colors(img):
    shape_ = img.shape[-1]
    return img * np.random.normal(1.0, 0.1, size=(1, 1, shape_)) + np.random.normal(
        0.0, 0.1, size=(1, 1, shape_)
    )


class Shift_Colors(ImageOnlyTransform):
    def apply(self, img, **params):
        return shift_colors(img)

    def get_transform_init_args_names(self):
        return ()


def create_umap_embeddings(
    feature_bank,
    targets,
    id_bank=None,
    label_mapper=None,
    label_mapper_id_bank=None,
    umap_path="umap_emb",
    umap_path_id_bank="umap_emb_id_bank",
    metadata=[]
):

    if not isinstance(targets, np.ndarray):
        targets = np.asarray(targets)

    if label_mapper is None:
        n_classes = len(np.unique(targets))
        label_mapper = {val: "class_" + str(val) for val in range(n_classes)}
    named_targets = np.asarray([label_mapper[trgt] for trgt in targets])

    # create umaps and plot
    # idx = random.randrange(named_targets.shape[0])
    embedding = umap.UMAP().fit(feature_bank)  # [idx])

    #     emb = embedding.transform()

    #     print("named_targets: ", named_targets)
    # figure_handler = umap.plot.points(embedding,
    # points=embedding.transform(feature_bank), labels=named_targets)#[idx])
    figure_handler = umap.plot.points(embedding, labels=named_targets)
    plt.savefig(umap_path, dpi=300)
    plt.close()
    if id_bank is not None:

        # pdb.set_trace()

        id_bank = np.concatenate(id_bank)
        if not isinstance(id_bank, np.ndarray):
            id_bank = np.asarray(id_bank)

        if True:
            plot_emb = embedding.embedding_[(id_bank != 0).astype(bool)]
            plot_lab = id_bank[(id_bank != 0).astype(bool)]
            plt.figure()
            plt.scatter(plot_emb[:, 0], plot_emb[:, 1], s=5.0, c=plot_lab, cmap="hsv")
            # plt.legend(fontsize=10)
            umap_df = pd.DataFrame(plot_emb, columns=['UMAP1', 'UMAP2'])
#             print("umap_path_id_bank: ", umap_path_id_bank, umap_path)
            umap_df.to_csv(umap_path_id_bank[:-4] + ".csv")
            print("embeds saved to: ", umap_path_id_bank[:-4] + ".csv")
            plt.savefig(umap_path_id_bank, dpi=300)
            plt.close()

        else:
            if label_mapper_id_bank is None:
                n_classes = len(np.unique(id_bank))
                label_mapper_id_bank = {
                    val: "class_" + str(val) for val in range(n_classes)
                }

            named_targets = np.asarray([label_mapper_id_bank[trgt] for trgt in id_bank])
            figure_handler = umap.plot.points(embedding, named_targets)  # [idx])
            # plt.legend(fontsize=10)
            plt.savefig(umap_path_id_bank, dpi=300)
            plt.close()


class MultiCrop:
    def __init__(self, n_crops, sizes, scales, albumentations=False):
        if albumentations:
            _crop_fn = A.RandomResizedCrop
        else:
            _crop_fn = RandomResizedCrop
        self.n_crops = n_crops
        self.sizes = sizes
        self.scales = scales
        self.crop_augs = []
        print("-------------- USING ALBUMENTATIONS --------------")
        for n, s in enumerate(n_crops):
            size = sizes[n]
            scale = scales[n]
            for _ in range(s):
                if albumentations:
                    self.crop_augs.append(
                        _crop_fn(
                            height=size,
                            width=size,
                            scale=scale,
                            interpolation=cv2.INTER_CUBIC,
                        )
                    )
                else:
                    self.crop_augs.append(
                        _crop_fn(size=size, scale=scale, interpolation=cv2.INTER_CUBIC)
                    )

    def __call__(self, image, augmentations):
        images = []
        for c_aug in self.crop_augs:
            images.append(augmentations(c_aug(image)))

        if not images:
            return image
        return images


def modules_are_equal(m1, m2):
    for (k1, w1), (k2, w2) in zip(m1.named_parameters(), m2.named_parameters()):
        if w1.data.ne(w2.data).sum() > 0:
            return False
    return True


def cospace(start, stop, num):
    steps = np.arange(num)
    return stop + 0.5 * (start - stop) * (1 + np.cos(np.pi * steps / len(steps)))


class CosineSchedulerWithWarmup:
    def __init__(
        self, base_value, final_value, iters, warmup_iters=0, warmup_init_val=None
    ):
        if warmup_init_val is None:
            warmup_init_val = base_value
        self.base_value = base_value
        self.final_value = final_value
        self.iters = iters
        self.warmup_iters = warmup_iters
        self.warmup_init_val = warmup_init_val

        warmup_sch = np.linspace(warmup_init_val, base_value, warmup_iters)
        core_sch = cospace(base_value, final_value, iters - warmup_iters)
        self.scheduler = np.concatenate((warmup_sch, core_sch))

        if not self.scheduler.size:
            self.scheduler = np.array([base_value])

    def __len__(self):
        return len(self.scheduler)

    def __call__(self, it):
        if it < len(self):
            return self.scheduler[it]
        else:
            warnings.warn(
                "Iteration number exceeds scheduler's def - Proceeding with last value",
                UserWarning,
            )
            return self.scheduler[-1]


class MixedLRScheduler:
    __name__ = "MixedLRScheduler"

    def __init__(self, schedulers, scheduler_types, steps_per_epoch):
        """
        Expectes a list of schedulers and a list of scheduler types with the same order
        The init function will update the schedulers' iterations etc with high priority for warmups
        The step will make a step (based on iteration!) with high priority for warmups
        """
        self.iteration_based = ["LinearWarmup", "OneCycleLR"]
        self.continuous_based = ["CosineAnnealingWarmRestarts"]
        self.epoch_based = ["MultiStepLR"]
        self.other_types = ["ReduceLROnPlateau", "CosineAnnealingLR"]
        self.wanrup_based = ["LinearWarmup"]
        self.accepted_types = self.iteration_based + self.epoch_based + self.other_types
        self.schedulers = schedulers
        self.scheduler_types = scheduler_types
        self.steps_per_epoch = steps_per_epoch
        self.iter = 0

        self.warmup_iters = [
            self.schedulers[sc].warmup_iters
            for sc, sctype in enumerate(self.scheduler_types)
            if sctype in self.wanrup_based
        ]
        self.warmup_iters = max(self.warmup_iters) if self.warmup_iters else 0

    def reset_iters():
        self.iter = 0

    def step(self, val_acc=None, val_loss=None):
        self.iter += 1
        for stype, sch in zip(self.scheduler_types, self.schedulers):
            if stype in self.iteration_based:
                sch.step()
            elif stype in self.continuous_based:
                sch.step(self.iter / self.steps_per_epoch)
            elif stype in self.epoch_based:
                if (self.iter + 1) % self.steps_per_epoch == 0:
                    sch.step()
            elif stype == "ReduceLROnPlateau":
                if (self.iter + 1) % self.steps_per_epoch == 0:
                    if sch.mode == "min":
                        sch.step(val_loss)
                    else:
                        sch.step(val_acc)
            elif stype == "CosineAnnealingLR":
                if self.iter > self.warmup_iters:
                    sch.step()
            elif stype is None:
                pass
            else:
                raise ValueError(f"{stype} is not a supported scheduler")


def show_unused_params(subnet):
    for name, param in subnet.named_parameters():
        if param.grad is None:
            print(name)


def cancel_gradients(subnet, attr):
    for n, p in subnet.named_parameters():
        if attr in n:
            p.grad = None


def download_cnn_weights(backbone_type, pretrained_type):

    checkpoints_dino = {
        "resnet50": "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
    }

    checkpoints = {"dino": checkpoints_dino}

    try:
        chpt = torch.hub.load_state_dict_from_url(
            checkpoints[pretrained_type][backbone_type], progress=True
        )
    except BaseException:
        raise ValueError(
            f"Pretrained weights for {backbone_type} with pretrained method {pretrained_type} not found."
        )

    return chpt
