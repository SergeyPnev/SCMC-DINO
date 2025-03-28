import os
import pdb
from .bases import *
from utils.transformers import *
from torch.cuda.amp import autocast


class Identity(nn.Module):
    """An identity function."""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Classifier(BaseModel):
    """A wrapper class that provides different CNN backbones.

    Is not intended to be used standalone. Called using the DefaultWrapper class.
    """

    def __init__(self, model_params):
        super().__init__()
        self.attr_from_dict(model_params)

        if hasattr(transformers, self.backbone_type):
            # self.transformers_params["in_chans"] = self.img_channels
            self.backbone = transformers.__dict__[self.backbone_type](
                **self.transformers_params, pretrained=self.pretrained
            )
            fc_in_channels = self.backbone.num_features

        elif hasattr(cnn_models, self.backbone_type):
            self.backbone = cnn_models.__dict__[self.backbone_type](
                pretrained=self.pretrained
            )
            # loading non-standard weights
            pretrained_type = (
                self.cnn_params.pretrained_type
                if hasattr(self, "cnn_params")
                else "supervised"
            )
            if self.pretrained and pretrained_type != "supervised":
                pre_cpt = download_cnn_weights(self.backbone_type, pretrained_type)
                missed_keys = self.backbone.load_state_dict(pre_cpt, strict=False)
                missing_head = set(missed_keys.missing_keys) == set(
                    ["fc.weight", "fc.bias"]
                )
                unexpected_keys = missed_keys.unexpected_keys == []
                is_ok = missing_head and unexpected_keys
                if not is_ok:
                    raise ValueError(
                        f"Found unexpected keysor keys are missing: {missed_keys}"
                    )
                print_ddp(f"\033[96m Using pretrained type: {pretrained_type}\033[0m")
            fc_in_channels = self.backbone.fc.in_features
        else:
            raise NotImplementedError
        # removing the fc layer from the backbone (which is manually added
        # below)
        self.backbone.fc = Identity()

        # modify stem and last layer
        self.fc = nn.Linear(fc_in_channels, self.n_classes).cuda()
        self.modify_first_layer(self.img_channels, self.pretrained)

        # add aux fc head
        self.aux_fc = None
        if self.use_aux_fc:
            self.aux_fc = AuxModel(fc_in_channels, self.n_classes)

        # replaciing BN with other norm if the model is CNN-based
        if self.replace_BN["apply"] and hasattr(cnn_models, self.backbone_type):
            self.BN_to_OtherNorm(
                self.replace_BN["normtype"],
                num_groups=self.replace_BN["num_groups"],
                keep_stats=self.replace_BN["keep_stats"],
            )
        # reseting norm stats
        if self.reset_norm_stats:
            self.initialize_norm_layers()

        if self.freeze_backbone:
            self.freeze_submodel(self.backbone)

    def forward(self, x, return_embedding=False, calc_aux_out=True, patch_drop=0., return_attention=False):
        with autocast(self.use_mixed_precision):
            aux_out = None

            if self.freeze_backbone:
                self.backbone.eval()

            if isinstance(x, list) and hasattr(cnn_models, self.backbone_type):
                idx_crops = torch.cumsum(
                    torch.unique_consecutive(
                        torch.tensor([inp.shape[-1] for inp in x]),
                        return_counts=True,
                    )[1],
                    0,
                )
                start_idx = 0
                for end_idx in idx_crops:
                    _out = self.backbone(torch.cat(x[start_idx:end_idx]), patch_drop=patch_drop)
                    if start_idx == 0:
                        x_emb = _out
                    else:
                        x_emb = torch.cat((x_emb, _out))
                    start_idx = end_idx
            else:
                if return_attention:
                    x_emb, attn = self.backbone(x, patch_drop=patch_drop, return_attention=return_attention)
                else:
                    x_emb = self.backbone(x, patch_drop=patch_drop)

            x = self.fc(x_emb)
            if self.aux_fc is not None and calc_aux_out:
                aux_out = self.aux_fc(x_emb.clone().detach())

            if return_embedding:
                if not return_attention:
                    return x, x_emb, aux_out
                else:
                    return x, x_emb, aux_out, attn
            else:
                if not return_attention:
                    return x, aux_out
                else:
                    return x, aux_out, attn

    def forward_get_attention(self, x, return_embedding=False, calc_aux_out=True):

        with autocast(self.use_mixed_precision):
            aux_out = None

            if self.freeze_backbone:
                self.backbone.eval()

            if isinstance(x, list) and hasattr(cnn_models, self.backbone_type):
                idx_crops = torch.cumsum(
                    torch.unique_consecutive(
                        torch.tensor([inp.shape[-1] for inp in x]),
                        return_counts=True,
                    )[1],
                    0,
                )
                start_idx = 0
                for end_idx in idx_crops:
                    _out, attn = self.backbone.forward_get_last_attention(
                        torch.cat(x[start_idx:end_idx])
                    )
                    if start_idx == 0:
                        x_emb = _out
                    else:
                        x_emb = torch.cat((x_emb, _out))
                    start_idx = end_idx
            else:
                x_emb, attn = self.backbone.forward_get_last_attention(x)

            x = self.fc(x_emb)
            if self.aux_fc is not None and calc_aux_out:
                aux_out = self.aux_fc(x_emb.clone().detach())

            if return_embedding:
                return x, x_emb, aux_out, attn
            else:
                return x, aux_out

    def modify_first_layer(self, img_channels, pretrained):
        backbone_type = self.backbone.__class__.__name__
        if img_channels == 3:
            return

        if backbone_type == "ResNet":
            conv_attrs = [
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "groups",
                "bias",
                "padding_mode",
            ]
            conv1_defs = {
                attr: getattr(self.backbone.conv1, attr) for attr in conv_attrs
            }

            pretrained_weight = self.backbone.conv1.weight.data
            pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]

            self.backbone.conv1 = nn.Conv2d(img_channels, **conv1_defs)
            if pretrained:
                self.backbone.conv1.weight.data = pretrained_weight

        elif backbone_type == "VisionTransformer":
            patch_embed_attrs = ["img_size", "patch_size", "embed_dim"]
            patch_defs = {
                attr: getattr(self.backbone.patch_embed, attr)
                for attr in patch_embed_attrs
            }

            pretrained_weight = self.backbone.patch_embed.proj.weight.data
            if self.backbone.patch_embed.proj.bias is not None:
                pretrained_bias = self.backbone.patch_embed.proj.bias.data

            if not self.channel_agnostic:
                pretrained_weight = pretrained_weight.repeat(1, 4, 1, 1)[:, :img_channels]
                self.backbone.patch_embed = transformers.deit.PatchEmbed(
                    in_chans=img_channels, **patch_defs
                )
            else:
                pretrained_weight = pretrained_weight[:, :1]
                pos_embed_pretrained_weight = self.backbone.pos_embed.data.repeat(1, 5, 1)
                self.backbone.patch_embed = transformers.deit.ChannelAgnosticPatchEmbed(
                    in_chans=1, **patch_defs
                )

                self.backbone.pos_embed = self.backbone.generate_2d_sincos_pos_embeddings(
                    embedding_dim=self.backbone.num_features,
                    length=self.backbone.img_size/self.backbone.patch_size,
                    use_class_token=self.backbone.cls_token is not None,
                    num_modality=5
                )

            if pretrained:
                self.backbone.patch_embed.proj.weight.data = pretrained_weight
                if self.channel_agnostic:
                    self.backbone.pos_embed.data = pos_embed_pretrained_weight
                if self.backbone.patch_embed.proj.bias is not None:
                    self.backbone.patch_embed.proj.bias.data = pretrained_bias

        else:
            raise NotImplementedError(
                "channel modification is not implemented for {}".format(backbone_type)
            )


class AuxModel(BaseModel):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        self.net = nn.Linear(in_channels, n_classes)
        self.org_model_state = model_to_CPU_state(self.net)

    def forward(self, x):
        return self.net(x)

    def reset(self):
        self.net.load_state_dict(self.org_model_state)
        self.net.to(self.device_id)
