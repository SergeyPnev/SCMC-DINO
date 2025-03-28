import torch.nn as nn
import torch.nn.functional as F
from defaults.bases import *
from torch.cuda.amp import autocast
from pytorch_metric_learning import losses


class DINOHeadCells(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        n_cells,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256
    ):
        super().__init__()

        self.n_cells = n_cells
        self.in_dim = in_dim
        self.out_dim = out_dim

#         if self.n_cells > 1:
#             print('Aggregate embeddings using average.')

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        assert self.in_dim == x.shape[-1]
#         if self.n_cells > 1:
#             B = int(x.shape[0] / self.n_cells)
#             x = x.view(B, self.n_cells, self.in_dim)
#             x = torch.mean(x, dim=1)

        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOHead(nn.Module):
    # Taken from
    # https://github.com/facebookresearch/dino/blob/a52c63ba27ae15856a5d99e42c5a1b82daa902d8/vision_transformer.py#L314
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        norm_last_layer=True,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
#         print("feature_vectors: ", feature_vectors.shape)
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
#         print("feature_vectors_normalized: ", feature_vectors_normalized.shape)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))


class Model(BaseModel):
    def __init__(self, student):
        super().__init__()

        # self.n_classes = student.n_classes
        self.n_classes = 51
        # self.n_classes = 571
        self.n_domain_classes = student.n_domain_classes
        self.img_channels = student.img_channels
        self.num_domains = student.num_domains
        self.examples_in_each_domain = student.examples_in_each_domain
        in_dim = student.fc.in_features
        self.in_dim = in_dim

        dino_args = student.DINO if hasattr(student, "DINO") else {}
        projection_size = dino_args.get("projection_size", 2048)
        moving_average_decay = dino_args.get("moving_average_decay", 0.996)
        moving_average_decay_end = dino_args.get("moving_average_decay_end", 1.0)

        self.multi_center_training = dino_args.get("multi_center_training", True)
        self.embedding_centering = dino_args.get("embedding_centering", False)
        self.embedding_centering_lambda = dino_args.get(
            "embedding_centering_lambda", 0.5
        )

        self.update_centering_head = dino_args.get("update_centering_head", True)
        self.update_centering_features = dino_args.get(
            "update_centering_features", True
        )

        self.loss_fn = nn.CrossEntropyLoss()

        # create online and target encoders
        self.student_encoder = student
        self.student_encoder.aux_fc = None

        self.n_cells = dino_args.get("n_cells", 8)
        self.use_bn_in_head = dino_args.get("head_bn", True)
        self.norm_last_layer = dino_args.get("norm_last_layer", True)
        self.hidden_dim = dino_args.get("hidden_dim", 1024)
        self.bottleneck_dim = dino_args.get("bottleneck_dim", 1024)
        self.n_head_layers = dino_args.get("n_head_layers", 3)

        # create online projectors and predictors
        self.student_encoder.fc = nn.Identity()

        self.treatment_head = DINOHeadCells(
            in_dim=in_dim,
            out_dim=self.n_classes,
            n_cells=self.n_cells,
            use_bn=self.use_bn_in_head,
            norm_last_layer=self.norm_last_layer,
            hidden_dim=self.hidden_dim,
            bottleneck_dim=self.bottleneck_dim,
            nlayers=self.n_head_layers,
        )

#         self.domain_head = DINOHead(
#             in_dim=in_dim,
#             out_dim=self.n_domain_classes,
#             use_bn=self.use_bn_in_head,
#             norm_last_layer=self.norm_last_layer,
#             hidden_dim=self.hidden_dim,
#             bottleneck_dim=self.bottleneck_dim,
#             nlayers=self.n_head_layers,
#         )

        # self.criterion = SupervisedContrastiveLoss(0.1)
        self.criterion = nn.CrossEntropyLoss()

        # send the BYOL wrapped model to the original model's GPU ID
        self.softmax = nn.Softmax()
        self.to(self.device_id)

    def forward(self, x, labels=None, domain_labels=None, return_embedding=False, domain_belonging=None, epoch=0, it=0):
        # Forward pass
        aux_outs = None
#         print("len x: ", len(x))
#         print("x: ", x.shape)

        with autocast(self.use_mixed_precision):
            if return_embedding:
                x = x.to(self.device_id, non_blocking=True)
                student_out = self.student_encoder.backbone(x)
                print(student_out.shape)
                if self.n_cells > 1:
                    out = student_out.reshape((-1, self.n_cells, self.in_dim))
                    out = torch.mean(out, dim=1)

                if self.student_encoder.aux_fc is not None:
                    aux_outs = self.treatment_head(student_out)
                return None, out, aux_outs

            x = x.to(self.device_id)

            # all views pass through the student
            student_out, student_output_backbone, _ = self.student_encoder(x, return_embedding=True)

#             print("student_out:", len(student_out), len(domain_labels[0]))

            treatment_logits = self.treatment_head(student_output_backbone)
#             domain_logits = self.domain_head(student_out)

#             loss_treatment = self.loss_fn(
#                 treatment_logits,
#                 labels.to(self.device_id)
#             )

            loss = self.criterion(treatment_logits, labels.to(self.device_id))

        return loss.mean(), aux_outs
