import pdb
import wandb
from self_supervised.BYOL.trainer import *


class SupervisedTrainer(BYOLTrainer):
    def __init__(
        self, wraped_defs, freeze_last_for=1, final_weight_decay=0.4, stop_early=0
    ):
        super().__init__(wraped_defs, stop_early=stop_early)
        self.freeze_last_for = freeze_last_for
        self.stop_early = stop_early
        self.decay_scheduler = CosineSchedulerWithWarmup(
            base_value=self.optimizer.param_groups[0]["weight_decay"],
            final_value=final_weight_decay,
            iters=len(self.trainloader) * self.epochs,
        )

    def train(self):
        self.test_mode = False
        self.print_train_init()
        self.load_session(False)

        epoch_bar = range(self.epoch0 + 1, self.epoch0 + self.epochs + 1)
        if self.is_rank0:
            epoch_bar = tqdm(epoch_bar, desc="Epoch", leave=False)

#         self.epoch_step()
        # Looping thorough epochs
        for self.epoch in epoch_bar:
            if isinstance(self.trainloader.sampler, DS):
                self.trainloader.sampler.set_epoch(self.epoch)
            self.model.train()
            iter_bar = enumerate(self.trainloader)
            if self.is_rank0:
                iter_bar = tqdm(
                    iter_bar, desc="Training", leave=False, total=len(self.trainloader)
                )

            # Looping through batches
            if self.validate_learning_on_single_batch:
                for it, batch in iter_bar:
                    break

                for it in range(int(len(iter_bar))):
                    self.iters += 1
                    print(self.iters)
                    self.global_step(batch=batch, it=it)

                    # going through epoch step
                    if self.val_every != np.inf:
                        if self.iters % int(self.val_every * self.epoch_steps) == 0:
                            synchronize()
                            self.epoch_step()
                            self.model.train()
                    synchronize()

            else:
                for it, batch in iter_bar:
                    self.iters += 1
                    self.global_step(batch=batch, it=it)

                    # going through epoch step
                    if self.val_every != np.inf:
                        if self.iters % int(self.val_every * self.epoch_steps) == 0:
                            synchronize()
                            self.epoch_step()
                            self.model.train()
                    synchronize()

            if not self.save_best_model and not self.is_grid_search:
                self.best_model = model_to_CPU_state(self.feature_extractor)
                self.save_session()

            if self.use_aux_head and self.reset_aux_every:
                if (
                    self.epoch < self.epochs + 1
                    and self.epoch % self.reset_aux_every == 0
                ):
                    if is_ddp(self.model):
                        if self.model.module.student_encoder.aux_fc is not None:
                            self.model.module.student_encoder.aux_fc.reset()
                        if self.model.module.teacher_encoder.aux_fc is not None:
                            self.model.module.teacher_encoder.aux_fc.reset()
                    else:
                        if self.model.student_encoder.aux_fc is not None:
                            self.model.student_encoder.aux_fc.reset()
                        if self.model.teacher_encoder.aux_fc is not None:
                            self.model.teacher_encoder.aux_fc.reset()
            print(self.epoch, self.stop_early)
            if self.stop_early > 0 and self.stop_early == self.epoch:
                break

        print_ddp(" ==> Training done")
        if not self.is_grid_search:
            self.save_session(verbose=True)
        synchronize()

    def global_step(self, **kwargs):
        self.optimizer.zero_grad()
        self.aux_optimizer.zero_grad()
        aux_loss = None

        # get batch
        images, labels = kwargs["batch"]
        ids = labels[-1]
        plate = labels[0][1]
        moa = labels[0][2]
        split = labels[0][3]
        labels = labels[0][0]

        labels = moa

        # go through the model
        with autocast(self.use_mixed_precision):
            loss, aux_outs = self.model(
                images, labels, ids
            )
        # backprop
        if aux_outs is not None:
            aux_loss = self.criterion(
                aux_outs, labels.to(self.device_id, non_blocking=True)
            )
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
            if self.epoch <= self.freeze_last_for:
                cancel_gradients(self.model, "student_encoder.fc.last_layer")
            self.optimizer.step()
        else:
            self.scaler.scale(loss).backward()
            if self.grad_clipping:
                self.scaler.unscale_(self.optimizer)
                clipped_params = (
                    w for n, w in self.model.named_parameters() if "aux_fc" not in n
                )
                torch.nn.utils.clip_grad_norm_(clipped_params, self.grad_clipping)
            if self.epoch <= self.freeze_last_for:
                cancel_gradients(self.model, "student_encoder.fc.last_layer")
            self.scaler.step(self.optimizer)
            self.scaler.update()

        # updating lr and wd
        self.scheduler.step(self.val_target, self.val_loss)
        if aux_outs is not None and self.aux_scheduler is not None:
            self.aux_scheduler.step(self.val_target, self.val_loss)
        self.optimizer.param_groups[0]["weight_decay"] = self.decay_scheduler(
            self.iters
        )
        if self.iters % self.log_every == 0 or (
            self.iters == 1 and not self.is_grid_search
        ):
            loss = dist_average_tensor(loss)
            if self.is_rank0:
                log_dict = {"train_loss": loss.item(), "learning_rate": self.get_lr()}
                if aux_outs is not None:
                    log_dict["aux_learning_rate"] = self.aux_optimizer.param_groups[0][
                        "lr"
                    ]
                self.logging(log_dict)
                if aux_loss is not None:
                    self.logging({"aux_loss": aux_loss.item()})

    def epoch_step(self, **kwargs):
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
        val_taget="knn",
        **kwargs,
    ):
        """Validation loop function.
        This is pretty much the same thing with global_step() but with torch.no_grad()
        Also note that DDP is not used here. There is not much point to DDP, since
        we are not doing backprop anyway.
        """
        used_val_for_best = f"{val_taget}_{prefix}"
        if knn_eval and calc_train_feats:
            self.build_feature_bank()

        if not self.is_rank0:
            return
        # Note: I am removing DDP from evaluation since it is slightly slower
        self.model.eval()
        if dataloader is None:
            dataloader = self.valloader

        if not len(dataloader):
            self.best_model = model_to_CPU_state(self.feature_extractor)
            self.model.train()
            return

        n_classes = dataloader.dataset.n_classes
        n_domains = dataloader.dataset.num_domains
        knn_nhood = dataloader.dataset.knn_nhood
        target_metric = dataloader.dataset.target_metric

        aux_metric, knn_metric = None, None
        if self.is_rank0:
            if knn_eval:
                knn_metric = self.metric_fn(
                    n_domains, dataloader.dataset.domain_to_labels, mode=f"knn_{prefix}"
                )
            iter_bar = tqdm(
                dataloader, desc="Validating", leave=False, total=len(dataloader)
            )
        else:
            iter_bar = dataloader

        self.val_loss = None
        aux_val_loss = []
        feature_bank = []
        id_bank = []
        label_bank = []
        domain_bank = []
        moa_bank = []

        with torch.no_grad():
            ijk = 0
            for images, labels in iter_bar:

                ids = labels[1]
                plate = labels[1]
                domain = plate
                moa = labels[2]
                split = labels[3]
                labels = labels[0]

                # DOMAIN training
                labels = moa

                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)

                if is_ddp(self.model):
                    _, features, aux_outs = self.model.module(
                        images, labels, return_embedding=True
                    )
                else:
                    _, features, aux_outs = self.model(images, labels, return_embedding=True)

                if self.log_embeddings:
                    feature_bank.append(features.clone().detach().cpu())
                    id_bank.append(ids)
                    label_bank.append(labels)
                    domain_bank.append(domain)
                    moa_bank.append(ids)

                # knn_eval
                if knn_eval:
                    features = F.normalize(features, dim=1)
                    pred_labels = self.knn_predict(
                        feature=features,
                        feature_bank=self.feature_bank,
                        feature_labels=self.targets_bank.cuda(),
                        knn_k=knn_nhood,
                        knn_t=0.1,
                        classes=n_classes,
#                         classes=n_domains,
                        multi_label=not dataloader.dataset.is_multiclass,
                    )
                    knn_metric.add_preds(pred_labels, labels, using_knn=True)
                ijk += 1

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

        self.get_saved_model_path()
        base_path, model_name = self.model_path.split("checkpoints/")
        emb_path = model_name + "-{}".format("val")
        if self.iters >= 0:
            emb_path += "_iter{}".format(self.iters)
        emb_path += ".csv"
        emb_dir = os.path.join(base_path, "embeddings", model_name)
        embedding_path = os.path.join(emb_dir, emb_path)
        print(f"SAVED TO {embedding_path}")

        # building Umap embeddings
        if self.log_embeddings and knn_eval:
            self.build_umaps(
                feature_bank,
                dataloader,
                labels=knn_metric.truths,
                id_bank=id_bank,
                mode=f"{prefix}",
            )
            df.to_csv(embedding_path)

        eval_metrics = edict({})
        if knn_eval:
            knn_metric = knn_metric.get_value(use_dist=isinstance(dataloader, DS))
            eval_metrics.update(knn_metric)
            setattr(
                self,
                f"knn_{prefix}_target",
                knn_metric[f"knn_{prefix}_{target_metric}"],
            )

        if used_val_for_best == self.used_type_for_save_best:
            self.val_target = getattr(self, f"{used_val_for_best}_target")

        if not self.is_grid_search:
            if self.report_intermediate_steps:
                self.logging(eval_metrics)
            if self.val_target > self.best_val_target:
                self.best_val_target = self.val_target
                if self.save_best_model:
                    self.best_model = model_to_CPU_state(self.feature_extractor)
            if not self.save_best_model:
                self.best_model = model_to_CPU_state(self.feature_extractor)
        self.model.train()

    @property
    def feature_extractor(self):
        return DINO_to_classifier(self.model)

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
        n_domains = dataloader.dataset.num_domains
        target_metric = dataloader.dataset.target_metric

        aux_metric, knn_metric = None, None

        if self.is_supervised:
#         DOMAIN training
            metric = self.metric_fn(
                n_domains, dataloader.dataset.domain_to_labels, mode="test"
            )
        if knn_eval or not self.is_supervised:
            knn_metric = self.metric_fn(
                n_domains, dataloader.dataset.domain_to_labels, mode="knn_test"
            )
        if (knn_eval or not self.is_supervised) and self.normalized_per_domain:
            norm_features_feature_bank = None
            knn_norm_metric = self.metric_fn(
                n_domains, dataloader.dataset.domain_to_labels, mode="knn_norm_test"
            )
        iter_bar = tqdm(dataloader, desc="Testing", leave=True, total=len(dataloader))

        self.model.eval()
        with torch.no_grad():
            for images, labels in iter_bar:

                ids = labels[1]
                domain = labels[1]
                moa = labels[2]
                split = labels[3]
                labels = labels[0]

                labels = moa

                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)

                if is_ddp(self.model):
                    outputs, features, aux_outs = self.model.module(
                        images, labels, return_embedding=True
                    )
                else:
                    outputs, features, aux_outs = self.model(
                        images, labels, return_embedding=True
                    )

                if self.log_embeddings:
                    feature_bank.append(features)
                    id_bank.append(ids)
                    label_bank.append(labels)
                    domain_bank.append(domain)
                    moa_bank.append(ids)

                if knn_eval:
                    features = F.normalize(features, dim=1)
                    pred_labels = self.knn_predict(
                        feature=features,
                        feature_bank=self.feature_bank,
                        feature_labels=self.targets_bank.cuda(),
                        knn_k=knn_nhood,
                        knn_t=0.1,
                        classes=n_domains,
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
#                             classes=n_classes,
                            classes=n_domains,
                            multi_label=not dataloader.dataset.is_multiclass,
                        )
                        knn_norm_metric.add_preds(
                            pred_labels_norm, labels, using_knn=True
                        )

                if self.is_supervised:
                    loss = self.criterion(outputs, labels)
                    test_loss.append(loss.item())
                    metric.add_preds(outputs, labels)

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

            print("logging...")

        print("store_embeddings: ", store_embeddings)
        if store_embeddings:
            print("storring!")
            feature_bank = torch.cat(feature_bank, dim=0).detach().cpu().numpy()

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

            # id_bank = [item.decode("utf-8") for sublist in id_bank for item in sublist]

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

            print("SAVING JSON TO: ", metrics_path)
            self.model.train()
            self.set_models_precision(self.use_mixed_precision)
            save_json(test_metrics, metrics_path)

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
        n_domains = dataloader.dataset.num_domains
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
                ids = labels[1]
                plate = labels[1]
                moa = labels[2]
                split = labels[3]
                labels = labels[0]

                # DOMAIN training
                labels = moa

                ids_available = True

                images = images.to(self.device_id, non_blocking=True)

                outputs, feature, aux_outs = self.model(
                    images, return_embedding=True
                )

                # only invariant part!
                feature = F.normalize(feature, dim=1)
                self.feature_bank.append(feature)
                self.targets_bank.append(labels.clone().detach().cpu())
                domain_bank.append(ids.detach().cpu())
                moa_bank.append(moa.detach().cpu())
                split_bank.append(split)

                if ids_available:
                    self.id_bank.append(ids)

            self.feature_bank = torch.cat(self.feature_bank, dim=0).t().contiguous()
            self.targets_bank = torch.cat(self.targets_bank, dim=0).t().contiguous()
            domain_bank_np = torch.cat(domain_bank, dim=0).numpy()
            moa_bank_np = torch.cat(moa_bank, dim=0).numpy()
            split_bank_np = torch.cat(split_bank, dim=0).numpy()

            df = pd.DataFrame(
                self.feature_bank.t().contiguous().detach().cpu().numpy(),
                columns=["feature_" + str(ind_) for ind_ in range(self.feature_bank.t().contiguous().shape[1])],
            )
            df["moa"] = moa_bank_np
            df["label"] = self.targets_bank.t().detach().cpu().numpy()
            df["plate"] = domain_bank_np
            df["split"] = split_bank_np

            self.get_saved_model_path()
            base_path, model_name = self.model_path.split("checkpoints/")
            emb_path = model_name + "-{}".format("train")
            if self.iters >= 0:
                emb_path += "_iter{}".format(self.iters)
            emb_dir = os.path.join(base_path, "embeddings", model_name)
            embedding_path = os.path.join(emb_dir, emb_path + ".csv")
            embedding_path_dino = os.path.join(emb_dir, emb_path + "_dino.csv")
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

def DINO_to_classifier(net):
    if is_parallel(net):
        return net.module.student_encoder
    else:
        return net.student_encoder
