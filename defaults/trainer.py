import os
import pdb
import wandb
import defaults
from defaults.bases import *
import matplotlib.pyplot as plt
from .wrappers import DefaultWrapper, dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler as DS
from scipy.stats import chi2


class Trainer(BaseTrainer):
    """Main trainer class.

    Initializes with a DefaultWrapper instance as its input.
    Call trainer.train() to train and validate or call trainer.test()
    Training with standard DDP: a model is trainedon multiple machines/gpus using distributed gradients.
    """

    def __init__(self, wraped_defs):
        """Initialize the trainer instance.

        This function clones its attributes from the DefaultWrapper instance or generates
        them from the .json file.
        """
        super().__init__()

        self.parameters = wraped_defs.parameters
        self.is_supervised = wraped_defs.is_supervised
        self.training_params = self.parameters.training_params
        self.attr_from_dict(self.training_params)
        self.attr_from_dict(wraped_defs.dataloaders)
        self.epoch_steps = len(self.trainloader)
        self.total_steps = int(len(self.trainloader) * self.epochs)

        self.model = wraped_defs.model
        self.criterion = wraped_defs.criterion
        self.optimizer = wraped_defs.optimizer
        self.aux_scheduler, self.aux_optimizer = None, None
        if wraped_defs.aux_optimizer is not None:
            self.aux_optimizer = wraped_defs.aux_optimizer
        if wraped_defs.aux_schedulers is not None:
            self.aux_scheduler = wraped_defs.aux_schedulers
        self.scheduler = wraped_defs.schedulers
        self.metric_fn = wraped_defs.metric

        self.org_model_state = model_to_CPU_state(self.model)
        self.org_optimizer_state = opimizer_to_CPU_state(self.optimizer)
        self.total_step = len(self.trainloader)
        self.best_model = deepcopy(self.org_model_state)

        if self.use_mixed_precision:
            self.scaler = GradScaler()
            self.set_models_precision(self.use_mixed_precision)

    def train(self):
        """Main training loop."""
        self.test_mode = False
        self.load_session(False)
        self.print_train_init()

        n_classes = self.trainloader.dataset.n_classes
        metric = self.metric_fn(
            n_classes, self.trainloader.dataset.int_to_labels, mode="train"
        )
        epoch_bar = range(self.epoch0 + 1, self.epoch0 + self.epochs + 1)
        if self.is_rank0:
            epoch_bar = tqdm(epoch_bar, desc="Epoch", leave=False)

        for self.epoch in epoch_bar:
            self.model.train()
            if isinstance(self.trainloader.sampler, DS):
                self.trainloader.sampler.set_epoch(self.epoch)

            iter_bar = enumerate(self.trainloader)
            if self.is_rank0:
                iter_bar = tqdm(
                    iter_bar, desc="Training", leave=False, total=len(self.trainloader)
                )
            for it, batch in iter_bar:
                self.iters += 1
                self.global_step(batch=batch, metric=metric, it=it)
                if self.val_every != np.inf:
                    if self.iters % int(self.val_every * self.epoch_steps) == 0:
                        synchronize()
                        self.epoch_step()
                        self.model.train()
                synchronize()

            if not self.save_best_model and not self.is_grid_search:
                self.best_model = model_to_CPU_state(self.model)
                self.save_session()

            if self.use_aux_head and self.reset_aux_every:
                if (
                    self.epoch < self.epochs + 1
                    and self.epoch % self.reset_aux_every == 0
                ):
                    self.model.aux_fc.reset()

        if self.is_rank0:
            print(" ==> Training done")
        if not self.is_grid_search:
            self.evaluate()
            self.save_session(verbose=True)
        synchronize()

    def global_step(self, **kwargs):
        """Function for the standard forward/backward/update.

        If using DDP, metrics (e.g. accuracy) are calculated with dist.all_gather
        """
        self.optimizer.zero_grad()
        self.aux_optimizer.zero_grad()
        aux_loss = None
        metric = kwargs["metric"]
        images, labels = kwargs["batch"]
        if len(labels) == 2 and isinstance(labels, list):
            ids = labels[1]
            labels = labels[0]

        labels = labels.to(self.device_id, non_blocking=True)
        images = images.to(self.device_id, non_blocking=True)

        if labels.isnan().sum() or images.isnan().sum():
            print("problem ")

        with autocast(self.use_mixed_precision):
            outputs, aux_outs = self.model(images)
            loss = self.criterion(outputs)

        if loss.isnan().sum().cpu() != 0:
            print("Loss problem ", loss)

        if aux_outs is not None:
            aux_loss = self.criterion(aux_outs, labels)
            if not self.use_mixed_precision:
                aux_loss.backward()
                self.aux_optimizer.step()
            else:
                self.scaler.scale(aux_loss).backward()
                self.scaler.step(self.aux_optimizer)
                self.scaler.update()
        if not self.use_mixed_precision:
            loss.backward()
            if self.grad_clipping:
                clipped_params = (
                    w for n, w in self.model.named_parameters() if "aux_fc" not in n
                )
                torch.nn.utils.clip_grad_norm_(clipped_params, self.grad_clipping)
            self.optimizer.step()
        else:
            self.scaler.scale(loss).backward()
            if self.grad_clipping:
                self.scaler.unscale_(self.optimizer)
                clipped_params = (
                    w for n, w in self.model.named_parameters() if "aux_fc" not in n
                )
                torch.nn.utils.clip_grad_norm_(clipped_params, self.grad_clipping)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        metric.add_preds(outputs, labels)  # distributed gather inside
        self.scheduler.step(self.val_target, self.val_loss)
        if aux_outs is not None and self.aux_scheduler is not None:
            self.aux_scheduler.step(self.val_target, self.val_loss)

        if not self.is_grid_search:

            if self.iters % self.log_every == 0 or self.iters == 1:
                loss = dist_average_tensor(loss)

                if self.is_rank0:
                    log_dict = {
                        "train_loss": loss.item(),
                        "learning_rate": self.get_lr(),
                    }
                    if aux_outs is not None:
                        log_dict["aux_learning_rate"] = self.aux_optimizer.param_groups[
                            0
                        ]["lr"]
                    if self.is_rank0:
                        self.logging(log_dict)
                        self.logging(metric.get_value())
                        metric.reset()

    def epoch_step(self, **kwargs):
        """Function for periodic validation, LR updates and model saving.

        Note that in the 2nd phase of training, the behavior is different, each model on
        each GPU is saved separately.
        """
        self.val_iters += 1
        knn_eval = self.knn_eval_every and (self.val_iters % self.knn_eval_every) == 0
        self.evaluate(knn_eval=knn_eval)

        if not self.is_grid_search:
            self.save_session()

    def evaluate(
        self,
        dataloader=None,
        knn_eval=False,
        prefix="val",
        calc_train_feats=True,
        **kwargs,
    ):
        """Validation loop function.

        This is pretty much the same thing with global_step() but with torch.no_grad()
        Also note that DDP is not used here. There is not much point to DDP, since
        we are not doing backprop anyway.
        """

        # if we are using knn evaluations build a feature bank of the train set
        if knn_eval and calc_train_feats:
            self.build_feature_bank()

        if not self.is_rank0:
            return
        # Note: I am removing DDP from evaluation since it is slightly slower
        self.model.eval()

        if dataloader is None:
            dataloader = self.valloader

        if not len(dataloader):
            self.best_model = model_to_CPU_state(self.model)
            self.model.train()
            return

        knn_nhood = dataloader.dataset.knn_nhood
        n_classes = dataloader.dataset.n_classes
        target_metric = dataloader.dataset.target_metric

        aux_metric, knn_metric = None, None

        if self.is_rank0:
            metric = self.metric_fn(
                n_classes, dataloader.dataset.int_to_labels, mode=f"{prefix}"
            )
            if knn_eval:
                knn_metric = self.metric_fn(
                    n_classes, dataloader.dataset.int_to_labels, mode=f"knn_{prefix}"
                )
            if self.use_aux_head:
                aux_metric = self.metric_fn(
                    n_classes, dataloader.dataset.int_to_labels, mode=f"aux_{prefix}"
                )
            iter_bar = tqdm(
                dataloader, desc="Validating", leave=False, total=len(dataloader)
            )
        else:
            iter_bar = dataloader

        val_loss = []
        aux_val_loss = []
        feature_bank = []
        id_bank = []

        label_bank = []
        domain_bank = []
        moa_bank = []

        with torch.no_grad():

            for images, labels in iter_bar:
                if len(labels) == 2 and isinstance(labels, list):
                    ids = labels[1]
                    labels = labels[0]
                elif len(labels) == 3 and isinstance(labels, list):
                    ids = labels[2]
                    domain = labels[1]
                    labels = labels[0]

                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)

                if is_ddp(self.model):
                    outputs, features, aux_outs = self.model.module(
                        images, return_embedding=True
                    )
                else:
                    outputs, features, aux_outs = self.model(
                        images, return_embedding=True
                    )

                if self.log_embeddings:
                    feature_bank.append(features.clone().detach().cpu())
                    id_bank.append(ids)
                    label_bank.append(labels)
                    domain_bank.append(domain)
                    moa_bank.append(ids)

                if knn_eval:
                    features = F.normalize(features, dim=1)
                    pred_labels = self.knn_predict(
                        feature=features,
                        feature_bank=self.feature_bank,
                        feature_labels=self.targets_bank,
                        knn_k=knn_nhood,
                        knn_t=0.1,
                        classes=n_classes,
                        multi_label=not dataloader.dataset.is_multiclass,
                    )
                    knn_metric.add_preds(pred_labels, labels, using_knn=True)

                loss = self.criterion(outputs, labels)
                val_loss.append(loss.item())
                metric.add_preds(outputs, labels)

                if aux_metric is not None:
                    aux_loss = self.criterion(aux_outs, labels)
                    aux_val_loss.append(aux_loss.item())
                    aux_metric.add_preds(aux_outs, labels)

        feature_bank = torch.cat(feature_bank, dim=0).numpy()
        id_bank_np = torch.cat(domain_bank, dim=0).numpy()
        label_bank_np = torch.cat([lb.cpu() for lb in label_bank], dim=0).numpy()
        moa_bank_np = torch.cat(moa_bank, dim=0).numpy()

        df = pd.DataFrame(
            feature_bank,
            columns=["feature_" + str(ind_) for ind_ in range(feature_bank.shape[1])],
        )
        df["moa"] = moa_bank_np
        df["label"] = label_bank_np
        df["plate"] = id_bank_np

        embedding_path = (
            self.get_embedding_path(mode=mode, iters=self.iters) + f"/val_df_last.csv"
        )
        df.to_csv(embedding_path)
        print(
            f"SAVED TO {self.get_embedding_path(mode=mode, iters=self.iters) + f'/val_df_last.csv'}"
        )

        # building Umap embeddings
        if self.log_embeddings and self.make_umap_eval and False:
            self.build_umaps(
                feature_bank,
                dataloader,
                labels=metric.truths,
                id_bank=id_bank,
                mode=f"{prefix}",
            )

        self.val_loss = np.array(val_loss).mean()
        if aux_metric is not None:
            aux_val_loss = np.array(aux_val_loss).mean()
        eval_metrics = metric.get_value(use_dist=isinstance(dataloader, DS))
        if knn_eval:
            eval_metrics.update(
                knn_metric.get_value(use_dist=isinstance(dataloader, DS))
            )
        if aux_metric is not None:
            eval_metrics.update(
                aux_metric.get_value(use_dist=isinstance(dataloader, DS))
            )
        self.val_target = eval_metrics[f"{prefix}_{target_metric}"]

        if not self.is_grid_search:
            if self.report_intermediate_steps:
                self.logging(eval_metrics)
                self.logging({f"{prefix}_loss": round(self.val_loss, 5)})
                if aux_metric is not None:
                    self.logging({f"aux_{prefix}_loss": round(aux_val_loss, 5)})
            if self.val_target > self.best_val_target:
                self.best_val_target = self.val_target
                if self.save_best_model:
                    self.best_model = model_to_CPU_state(self.model)
            if self.val_loss <= self.best_val_loss:
                self.best_val_loss = self.val_loss
            if not self.save_best_model:
                self.best_model = model_to_CPU_state(self.model)
        self.model.train()

    def test(
        self,
        dataloader=None,
        knn_eval=True,
        store_embeddings=False,
        log_during_test=True,
        **kwargs,
    ):
        """Test function.

        Just be careful you are not explicitly passing the wrong dataset here.
        Otherwise it will use the test set.
        """
        # if not self.is_rank0: return

        self.test_mode = True
        self.restore_session = True
        self.restore_only_model = True
        self.set_models_precision(False)
        try:
            self.load_session(self.restore_only_model)
        except BaseException:
            print(
                "\033[93mFull checkpoint not found... "
                "Proceeding with partial model (assuming transfer learning is ON)\033[0m"
            )
        self.model.eval()
        if dataloader == None:
            dataloader=self.testloader

        if knn_eval:
            self.build_feature_bank(in_test=True)

        results_dir = os.path.join(self.save_dir, "results", self.model_name)
        metrics_path = os.path.join(results_dir, "metrics_results.json")
        check_dir(results_dir)

        print("RESULTS DIR: ", results_dir)

        test_loss = []
        aux_test_loss = []
        feature_bank = []
        id_bank = []

        label_bank = []
        domain_bank = []
        moa_bank = []

        results = edict()
        knn_nhood = dataloader.dataset.knn_nhood
        n_classes = dataloader.dataset.n_classes
        target_metric = dataloader.dataset.target_metric

        aux_metric, knn_metric = None, None
        if self.is_supervised:
            metric = self.metric_fn(
                n_classes, dataloader.dataset.int_to_labels, mode="test"
            )
        if knn_eval or not self.is_supervised:
            knn_metric = self.metric_fn(
                n_classes, dataloader.dataset.int_to_labels, mode="knn_test"
            )
        if (knn_eval or not self.is_supervised) and self.normalized_per_domain:
            norm_features_feature_bank = None
            knn_norm_metric = self.metric_fn(
                n_classes, dataloader.dataset.int_to_labels, mode="knn_norm_test"
            )
        if self.use_aux_head or not self.is_supervised:
            aux_metric = self.metric_fn(
                n_classes, dataloader.dataset.int_to_labels, mode="aux_test"
            )
        iter_bar = tqdm(dataloader, desc="Testing", leave=True, total=len(dataloader))

        self.model.eval()
        with torch.no_grad():
            for images, labels in iter_bar:
                if len(labels) == 3 and isinstance(labels, list):
                    ids = labels[2]
                    domain = labels[1]
                    labels = labels[0]
                elif len(labels) == 2 and isinstance(labels, list):
                    ids = labels[1]
                    labels = labels[0]
                elif len(labels) == 4 and isinstance(labels, list):
                    ids = labels[2]
                    domain = labels[1]
                    labels = labels[0]

                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)

                if is_ddp(self.model):
                    outputs, features, aux_outs = self.model.module(
                        images, return_embedding=True
                    )
                else:
                    outputs, features, aux_outs = self.model(
                        images, return_embedding=True
                    )

                if self.log_embeddings:
                    feature_bank.append(features.clone().detach().cpu())
                    id_bank.append(ids)
                    label_bank.append(labels)
                    domain_bank.append(domain)
                    moa_bank.append(ids)

                if knn_eval:
                    features = F.normalize(features, dim=1)
                    pred_labels = self.knn_predict(
                        feature=features,
                        feature_bank=self.feature_bank,
                        feature_labels=self.targets_bank,
                        knn_k=knn_nhood,
                        knn_t=0.1,
                        classes=n_classes,
                        multi_label=not dataloader.dataset.is_multiclass,
                    )
                    knn_metric.add_preds(pred_labels, labels, using_knn=True)
                    if self.normalized_per_domain:
                        norm_features = self.normalize_features_per_domain(
                            features, domain
                        )
                        norm_features = F.normalize(norm_features, dim=1)

                        if norm_features_feature_bank is None:
                            norm_features_feature_bank = (
                                self.normalize_features_per_domain(
                                    self.feature_bank.t(), self.id_bank
                                )
                            )
                            norm_features_feature_bank = F.normalize(
                                norm_features_feature_bank, dim=1
                            )
                            norm_features_feature_bank = norm_features_feature_bank.t()
                        pred_labels_norm = self.knn_predict(
                            feature=norm_features.cpu(),
                            feature_bank=norm_features_feature_bank.cpu(),
                            feature_labels=self.targets_bank.cpu(),
                            knn_k=knn_nhood,
                            knn_t=0.1,
                            classes=n_classes,
                            multi_label=not dataloader.dataset.is_multiclass,
                        )
                        knn_norm_metric.add_preds(
                            pred_labels_norm, labels, using_knn=True
                        )

                if self.is_supervised:
                    loss = self.criterion(outputs, labels)
                    test_loss.append(loss.item())
                    metric.add_preds(outputs, labels)

                if aux_metric is not None:
                    aux_loss = self.criterion(aux_outs, labels)
                    aux_test_loss.append(aux_loss.item())
                    aux_metric.add_preds(aux_outs, labels)

        if self.log_kbet and self.log_embeddings:
            id_bank_kbet = torch.tensor(
                [item for sublist in id_bank for item in sublist]
            )
            feature_bank_kbet = torch.cat(feature_bank, dim=0)

            acceptance_rate_dict_FB, _, _ = self.kbet(
                self.feature_bank.cpu().t(), self.feature_bank.cpu(), self.id_bank.cpu()
            )

            acceptance_rate_dict_TS, _, _ = self.kbet(
                feature_bank_kbet.cpu(), feature_bank_kbet.cpu().t(), id_bank_kbet.cpu()
            )

            norm_features_feature_bank_kbet = self.normalize_features_per_domain(
                feature_bank_kbet, id_bank_kbet
            )
            norm_features_feature_bank_kbet = F.normalize(
                norm_features_feature_bank_kbet, dim=1
            )
            acceptance_rate_dict_TS_norm, _, _ = self.kbet(
                norm_features_feature_bank_kbet.cpu(),
                norm_features_feature_bank_kbet.cpu().t(),
                id_bank_kbet.cpu(),
            )

        if store_embeddings:
            feature_bank = torch.cat(feature_bank, dim=0).numpy()

            id_bank_np = torch.cat(domain_bank, dim=0).numpy()
            label_bank_np = torch.cat([lb.cpu() for lb in label_bank], dim=0).numpy()
            moa_bank_np = torch.cat(moa_bank, dim=0).numpy()

            df = pd.DataFrame(
                feature_bank,
                columns=[
                    "feature_" + str(ind_) for ind_ in range(feature_bank.shape[1])
                ],
            )
            df["moa"] = moa_bank_np
            df["label"] = label_bank_np
            df["plate"] = id_bank_np


            self.get_saved_model_path()
            base_path, model_name = self.model_path.split("checkpoints/")
            emb_path = model_name + "-{}".format("test")
            if self.iters >= 0:
                emb_path += "_iter{}".format(self.iters)
            emb_path += ".csv"
            emb_dir = os.path.join(base_path, "embeddings", model_name)
            embedding_path = os.path.join(emb_dir, emb_path)
            print(f"SAVED TO {embedding_path}")
            df.to_csv(embedding_path)

            evaluate_moa = True
            if evaluate_moa:
                closest = self.eval_moa(df, ["feature_" + str(ind_) for ind_ in range(feature_bank.shape[1])])

            store_training_embeddings = True
            if store_training_embeddings:
                feature_bank_train = self.feature_bank.t().cpu().numpy()

                id_bank_np    = self.id_bank.cpu().numpy()
                label_bank_np = self.targets_bank.cpu().numpy()

                df = pd.DataFrame(feature_bank_train, columns = ["feature_"+str(ind_) for ind_ in range(feature_bank_train.shape[1])])

                df["label"]      = label_bank_np
                df["plate"]      = id_bank_np

                self.get_saved_model_path()
                base_path, model_name = self.model_path.split("checkpoints/")
                emb_path = model_name + "-{}".format("train-test")
                if self.iters >= 0:
                    emb_path += "_iter{}".format(self.iters)
                emb_path += ".csv"
                emb_dir = os.path.join(base_path, "embeddings", model_name)
                embedding_path = os.path.join(emb_dir, emb_path)
                print(f"SAVED TO {embedding_path}")

                df.to_csv(embedding_path)
                print("SAVED TO CSV")

            self.test_loss = np.array(test_loss).mean() if test_loss else None
            test_metrics = edict({})
            print('\n',"--"*5, f"{self.model_name} evaluated on the test set", "--"*5,'\n', "--"*28)
            if self.is_supervised:
                metric = metric.get_value(use_dist=isinstance(dataloader,DS))
                test_metrics.update(metric)
                self.print_results(test_metrics)
                if log_during_test:
                    self.logging(test_metrics)

            if knn_metric is not None:
                knn_metric = knn_metric.get_value(use_dist=isinstance(dataloader,DS))
                if self.log_kbet and self.log_embeddings:
                    res_FB = {"FB_" + str(key): val for key, val in acceptance_rate_dict_FB.items()}
                    res_TE = {"TE_" + str(key): val for key, val in acceptance_rate_dict_TS.items()}
                    res_TN = {"TN_" + str(key): val for key, val in acceptance_rate_dict_TS_norm.items()}

                    knn_metric = {**knn_metric, **res_FB, **res_TE, **res_TN}
                    if evaluate_moa:
                        knn_metric["moa-1-NN"] = closest[0]


                test_metrics.update(knn_metric)
                self.print_results(knn_metric)
                if log_during_test:
                    self.logging(knn_metric)

            if knn_norm_metric is not None:
                knn_norm_metric = knn_norm_metric.get_value(use_dist=isinstance(dataloader,DS))
                test_metrics.update(knn_norm_metric)
                self.print_results(knn_norm_metric)
                if log_during_test:
                    self.logging(knn_norm_metric)

            if aux_metric is not None:
                aux_metric = aux_metric.get_value(use_dist=isinstance(dataloader,DS))
                aux_test_loss = np.array(aux_test_loss).mean()
                aux_metric['aux_test_loss'] = round(aux_test_loss, 5)
                test_metrics.update(aux_metric)
                self.print_results(aux_metric)
                if log_during_test:
                    self.logging(aux_metric)

            print("SAVING JSON TO: ", metrics_path)
            self.model.train()
            self.set_models_precision(self.use_mixed_precision)
            save_json(test_metrics, metrics_path)

    def print_results(self, metric):
        print(
            tabulate(
                pd.DataFrame.from_dict(metric, orient="index").T,
                headers="keys",
                tablefmt="psql",
            )
        )
        print("\n", "--" * 35, "\n")

    def build_umaps(
        self,
        feature_bank,
        dataloader,
        labels=None,
        id_bank=None,
        mode="",
        wandb_logging=True,
    ):
        if not dataloader.dataset.is_multiclass:
            return
        addition_name = "_"
        # or feature_bank.type() != 'torch.FloatTensor':
        if isinstance(feature_bank, list):
            feature_bank = torch.cat(feature_bank, dim=0).numpy()
        else:
            addition_name = "normalized"

        umap_path = self.get_embedding_path(mode=mode, iters=self.iters)
        umap_path_id_bank = None
        if id_bank is not None:
            umap_path_id_bank = self.get_embedding_path(
                mode=mode + "_id_bank", iters=self.iters
            )

        create_umap_embeddings(
            feature_bank,
            labels,
            id_bank=id_bank,
            label_mapper=dataloader.dataset.int_to_labels,
            label_mapper_id_bank=dataloader.dataset.int_to_id,
            umap_path=umap_path,
            umap_path_id_bank=umap_path_id_bank,
        )

        if wandb_logging:
            if self.use_tensorboard:
                umap_plot = plt.imread(umap_path)
                self.logging({"umap_embeddings": [umap_plot[:, :, :3]]})
            else:
                umap_plot = Image.open(umap_path)
                if id_bank is not None:
                    umap_plot_id_bank = Image.open(umap_path_id_bank)
                    self.logging(
                        {
                            "umap_embeddings_"
                            + str(mode)
                            + addition_name: [
                                wandb.Image(umap_plot, caption=self.model_name)
                            ],
                            "umap_embeddings_id_bank"
                            + str(mode)
                            + addition_name: [
                                wandb.Image(
                                    umap_plot_id_bank,
                                    caption=self.model_name + "_id_bank",
                                )
                            ],
                        }
                    )
                else:
                    self.logging(
                        {
                            "umap_embeddings_"
                            + str(mode)
                            + addition_name: [
                                wandb.Image(umap_plot, caption=self.model_name)
                            ]
                        }
                    )

    def build_feature_bank(self, dataloader=None, in_test=False, **kwargs):
        """Build feature bank function.

        This function is meant to store the feature representation of the training images along with their respective labels

        This is pretty much the same thing with global_step() but with torch.no_grad()
        Also note that DDP is not used here. There is not much point to DDP, since
        we are not doing backprop anyway.
        """

        self.model.eval()
        if dataloader is None:
            dataloader = self.fbank_loader

        n_classes = dataloader.dataset.n_classes
        if self.is_rank0:
            iter_bar = tqdm(
                dataloader,
                desc="Building Feature Bank",
                leave=False,
                total=len(dataloader),
            )
        else:
            iter_bar = dataloader

        with torch.no_grad():

            self.feature_bank = []
            self.targets_bank = []
            self.id_bank = []
            split_bank = []
            domain_bank = []
            moa_bank = []
            ids_available = False
            for images, labels in iter_bar:
                moa, split = None, None
                if len(labels) == 2 and isinstance(labels, list):
                    ids = labels[1]
                    labels = labels[0]
                    ids_available = True
                if len(labels) == 3 and isinstance(labels, list):
                    ids = labels[1]
                    labels = labels[0]
                if len(labels) == 4 and isinstance(labels, list):
                    ids = labels[1]
                    moa = labels[2]
                    split = labels[3]
                    labels = labels[0]
                    ids_available = True
                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)
                if ids_available:
                    ids = ids.to(self.device_id, non_blocking=True)


                if is_ddp(self.model):
                    _, feature, _ = self.model.module(images, return_embedding=True)
                else:
                    _, feature, _ = self.model(images, return_embedding=True)

                feature = F.normalize(feature, dim=1)
                self.feature_bank.append(feature)
                self.targets_bank.append(labels)
                domain_bank.append(ids.detach().cpu())
                if moa is not None:
                    moa_bank.append(moa.detach().cpu())
                if split is not None:
                    split_bank.append(split)

                if ids_available:
                    self.id_bank.append(ids)

            self.feature_bank = torch.cat(self.feature_bank, dim=0).t().contiguous()
            self.targets_bank = torch.cat(self.targets_bank, dim=0).t().contiguous()
            domain_bank_np = torch.cat(domain_bank, dim=0).numpy()
            if moa is not None:
                moa_bank_np = torch.cat(moa_bank, dim=0).numpy()
            if split is not None:
                split_bank_np = torch.cat(split_bank, dim=0).numpy()

            df = pd.DataFrame(
                self.feature_bank.t().contiguous().detach().cpu().numpy(),
                columns=["feature_" + str(ind_) for ind_ in range(self.feature_bank.t().contiguous().shape[1])],
            )
            if moa is not None:
                df["moa"] = moa_bank_np
            df["label"] = self.targets_bank.t().detach().cpu().numpy()
            df["plate"] = domain_bank_np
            if split is not None:
                df["split"] = split_bank_np

            print(" label: ", self.targets_bank.t().detach().cpu().numpy())
            print(" plate: ", domain_bank_np.shape, domain_bank_np[:3])

            self.get_saved_model_path()
            base_path, model_name = self.model_path.split("checkpoints/")
            emb_path = model_name + "-{}".format("train")
            if self.iters >= 0:
                emb_path += "_iter{}".format(self.iters)
            emb_path += ".csv"
            emb_dir = os.path.join(base_path, "embeddings", model_name)
            embedding_path = os.path.join(emb_dir, emb_path)
            print(f"TRAIN EMBEDS SAVED TO {embedding_path}")
            os.makedirs(emb_dir, exist_ok=True)
            df.to_csv(embedding_path)

            if ids_available:
                self.id_bank = torch.cat(self.id_bank, dim=0).t().contiguous()

            if not in_test:
                synchronize()

            self.feature_bank = dist_gather(self.feature_bank, cat_dim=-1)
            self.targets_bank = dist_gather(self.targets_bank, cat_dim=-1)
            if ids_available:
                self.id_bank = dist_gather(self.id_bank, cat_dim=-1)

        self.model.train()

    def normalize_features_per_domain(self, features, domains):
        features = features.cpu()  # .numpy()
        domains = domains.cpu()  # .numpy()
        for dom in np.unique(domains):
            m = features[domains == dom].mean(axis=0)
            s = features[domains == dom].std(axis=0)
            features[domains == dom, :] = (features[domains == dom, :] - m) / (
                s + 1.0e-8
            )

        return features

    # FROM: Find the similarities between the batch samples and the feature
    # bank
    def knn_predict(
        self,
        feature,
        feature_bank,
        feature_labels,
        knn_k: int,
        knn_t: float,
        classes: int = 10,
        multi_label=False,
    ):
        """Helper method to run kNN predictions on features based on a feature bank

        Args:
            feature: Tensor of shape [N, D] consisting of N D-dimensional features
        feature_bank: Tensor of a database of features used for kNN
        feature_labels: Labels for the features in our feature_bank
        classes: Number of classes (e.g. 10 for CIFAR-10)
        knn_k: Number of k neighbors used for kNN
        knn_t:

        """

        # CHANGE TO FLAG
        if multi_label:
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            # Find the similarities between the batch samples and the feature
            # bank
            sim_matrix = torch.mm(feature, feature_bank)

            # identify the knn_k most similar samples in the feature bank for
            # each of the batch samples
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)

            # Expand the feature labels to a have a copy per batch sample
            expanded_labels = feature_labels.expand(
                (feature.size(0), feature_labels.size(0), feature_labels.size(1))
            )

            # Unsqueeze and expand the similarity indicies and weights
            sim_indices = sim_indices.unsqueeze_(1)
            sim_weight = sim_weight.unsqueeze_(1)
            sim_indices_expanded = sim_indices.expand(
                (sim_indices.size(0), feature_labels.size(0), sim_indices.size(2))
            )
            sim_weight_expanded = sim_weight.expand(
                (sim_weight.size(0), feature_labels.size(0), sim_weight.size(2))
            )

            # Gather the labels of the most similar samples in the feature bank
            gathered = torch.gather(expanded_labels, dim=-1, index=sim_indices_expanded)

            # Scale the weights of the most similar samples
            sim_weight_expanded = (sim_weight_expanded / knn_t).exp()

            # weight each of the labels
            weighted_labels = F.normalize(sim_weight_expanded, p=1, dim=2) * gathered
            pred_labels = weighted_labels.sum(axis=2)

            return pred_labels

        else:
            # compute cos similarity between each feature vector and feature
            # bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(
                feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
            )

            # we do a reweighting of the similarities
            sim_weight = (sim_weight / knn_t).exp()

            # counts for each class
            one_hot_label = torch.zeros(
                feature.size(0) * knn_k, classes, device=sim_labels.device
            )
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(
                dim=-1, index=sim_labels.view(-1, 1), value=1.0
            )
            # weighted score ---> [B, C]
            pred_scores = torch.sum(
                one_hot_label.view(feature.size(0), -1, classes)
                * sim_weight.unsqueeze(dim=-1),
                dim=1,
            )
            # convert them to probablilities
            pred_scores = pred_scores / pred_scores.sum(1).unsqueeze(1)
            # pred_labels = pred_scores.argsort(dim=-1, descending=True)[:, 0]

        return pred_scores

    def kbet(self, feature, feature_bank, feature_labels):
        """
        Calculate the kbet acceptance rate.
        """

        # compute cos similarity between each feature vector and feature bank
        # ---> [B, N]
        sim_matrix = torch.mm(feature, feature_bank)
        # [B, K]
        num_samples = feature_labels.shape[0]

        performance_dict = {}

        percentage = 0.005

        knn_k = int(percentage * num_samples)

        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)

        sim_indices = sim_indices[:, 1:]
        sim_weight = sim_weight[:, 1:]

        # [B, K]
        sim_labels = torch.gather(
            feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
        )
        count_list = []
        base_dist = []
        for val in feature_labels.unique():
            base_dist.append((feature_labels == val).sum())
            count_list.append((sim_labels == val).sum(axis=1))
        idea_dist = torch.stack(base_dist).t()
        idea_dist = idea_dist / idea_dist.sum()
        neighborhood = torch.stack(count_list).t()
        stat = (((neighborhood - idea_dist * knn_k) ** 2) / (idea_dist * knn_k)).sum(
            axis=1
        )
        dof = len(idea_dist) - 1
        p_value = 1 - chi2.cdf(stat, dof)
        acceptance_rate = (p_value >= 0.05).sum() / p_value.shape[0]

        performance_dict["acceptance_rate_knn_percentage_" + str(percentage)] = (
            acceptance_rate
        )

        return performance_dict, p_value, stat.mean()

    def eval_moa(self, df, embeds_cols):

        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score
        from collections import Counter

        def normalize_features_per_domain_control(
            features, domains, labels, control_id
        ):
            features = features  # numpy()
            domains = domains  # numpy()
            for dom in np.unique(domains):
                m = features[(domains == dom) & (labels == 0)].mean(axis=0)
                s = features[(domains == dom) & (labels == 0)].std(axis=0)
                features[domains == dom, :] = (features[domains == dom, :] - m) / (
                    s + 1.0e-8
                )

            return features

        def most_common(lst):
            return np.array(
                [Counter(sorted(row, reverse=True)).most_common(1)[0][0] for row in lst]
            )

        def NSC_k_NN(df_treatment, embeds_cols, plot_conf=False, savepath=None):
            # Create classes for each moa
            class_dict = dict(
                zip(
                    df_treatment["moa"].unique(),
                    np.arange(len(df_treatment["moa"].unique())),
                )
            )
            df_treatment["moa_class"] = df_treatment["moa"].map(class_dict)

            # Create nearest neighbors classifier
            predictions = list()
            labels = list()
            label_names = list()
            for comp in df_treatment["compound"].unique():
                df_ = df_treatment.loc[df_treatment["compound"] != comp, :]
                knn = KNeighborsClassifier(
                    n_neighbors=10, algorithm="brute", metric="cosine"
                )
                knn.fit(df_.loc[:, embeds_cols], df_.loc[:, "moa_class"])

                nn = knn.kneighbors(
                    df_treatment.loc[df_treatment["compound"] == comp, embeds_cols]
                )
                for p in range(nn[1].shape[0]):
                    predictions.append(list(df_.iloc[nn[1][p]]["moa_class"]))
                labels.extend(
                    df_treatment.loc[df_treatment["compound"] == comp, "moa_class"]
                )
                label_names.extend(
                    df_treatment.loc[df_treatment["compound"] == comp, "moa"]
                )

            predictions = np.asarray(predictions)

            k_nn_acc = [
                accuracy_score(labels, predictions[:, 0]),
                accuracy_score(labels, predictions[:, 1]),
                accuracy_score(labels, predictions[:, 2]),
                accuracy_score(labels, predictions[:, 3]),
                accuracy_score(labels, most_common(predictions[:, :1])),
                accuracy_score(labels, most_common(predictions[:, :2])),
                accuracy_score(labels, most_common(predictions[:, :3])),
                accuracy_score(labels, most_common(predictions[:, :4])),
                accuracy_score(labels, most_common(predictions[:, :5])),
                accuracy_score(labels, most_common(predictions[:, :10])),
            ]

            return k_nn_acc


        feature_columns = embeds_cols
        df["compound"] = df["label"]
        #         df[feature_columns] = normalize_features_per_domain_control(df[feature_columns].values, df["plate"].values, df["label"].values, 0)
        df[feature_columns] = F.normalize(
            torch.tensor(df[feature_columns].values), dim=1
        ).numpy()
        df[feature_columns] = df.groupby(["label"])[feature_columns].transform("mean")
        df_subset = df.drop_duplicates(subset=["label"])

        closest = NSC_k_NN(df_subset, feature_columns)

        return closest