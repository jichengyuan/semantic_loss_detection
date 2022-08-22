import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from .bbox_head import BBoxHead

import torch.nn.functional as F
from utils import binarize, pairwise_mahalanobis, CDs2Hg, HGNN
from torch.nn import init
from mmdet.models.losses import accuracy

@HEADS.register_module()
class HGNNBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none
                                    
                                                     /-> cls hist -> cls (only training)
                                    /-> cls convs -> 
                                                     \-> cls fcs  -> cls (training, validation and evaluation)
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(HGNNBBoxHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        # self.fc_out_channels = fc_out_channels
        self.fc_out_channels = fc_out_channels
        self.fc_in_features = 256
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dist_loss = .0

        self.d2hg = CDs2Hg(nb_classes=self.num_classes + 1,
                           sz_embed=512,
                           tau=32,
                           alpha=0.9)

        self.hnmp = HGNN(nb_classes=self.num_classes + 1,
                         sz_embed=512,
                         hidden=512)
        self.dist_weight = 1
        # HIST
        self.embedding_size = 512
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp_ = nn.AdaptiveMaxPool2d(1)
        self.embedding = nn.Linear(self.fc_in_features, self.embedding_size)
        self.cls_embedding = nn.Linear(self.embedding_size, self.num_classes + 1)

        self._initialize_weights()
        self.lnorm = nn.LayerNorm(self.embedding_size, elementwise_affine=False).cuda()

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area
        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                                                             self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    layer='Linear',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

    def _initialize_weights(self):

        # init.normal_(self.model.fc_cls.weight.data, 0,0.01)
        # init.constant_(self.model.fc_cls.bias, 0)
        init.kaiming_normal_(self.embedding.weight, mode='fan_out')
        init.constant_(self.embedding.bias, 0)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x, gt_labels=None, training=False):
        # shared part
        # if self.num_shared_convs > 0:
        #     for conv in self.shared_convs:
        #         x = conv(x)

        x_cls = x

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # separate branches
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        # if x_cls.dim() > 2:
        #     if self.with_avg_pool:
        #         x_cls = self.avg_pool(x_cls)
        #     x_cls = x_cls.flatten(1)
        # for fc in self.cls_fcs:
        #     x_cls = self.relu(fc(x_cls))

        x_cls_avg = self.gap(x_cls)
        x_cls_max = self.gmp_(x_cls)
        x_cls = x_cls_avg + x_cls_max
        x_cls = x_cls.view(x_cls.size(0), -1)
        x_cls = self.embedding(x_cls)
        x_embedding = self.lnorm(x_cls)
        cls_score = self.cls_embedding(x_embedding)
        # hist-head
        # gt_labels = gt_labels.detach().to("cuda:{}".format(x_embedding.get_device()))
        if training:
            self.dist_loss, H = self.d2hg(x_embedding, gt_labels)
            cls_score_hist = self.hnmp(x_embedding, H)
            cls_score = torch.cat((cls_score, cls_score_hist), dim=0)
            # cls_score_hist = self.fc_cls(x_cls_embedding) if self.with_cls else None
        #     x_cls = self.cls_embedding(x_embedding)

        # # cls-head
        # else:
        #     x_cls = self.cls_embedding(x_embedding)

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        # cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        return cls_score, bbox_pred


    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score.size(0) != bbox_pred.size(0):
            labels_cls = torch.cat((labels, labels), dim=0)
            label_weights_cls = torch.cat((label_weights, label_weights), dim=0)
        else:
            labels_cls = labels_cls
            label_weights_cls = label_weights_cls
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights_cls > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels_cls,
                    label_weights_cls,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)

                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_ + self.dist_weight * self.dist_loss
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels_cls)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels_cls)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute formaShared2HGNNBBoxHeadt.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

@HEADS.register_module()
class Shared2HGNNBBoxHead(HGNNBBoxHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2HGNNBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)