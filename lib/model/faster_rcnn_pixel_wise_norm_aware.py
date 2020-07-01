from collections import OrderedDict
import huepy as hue

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from torchvision.ops import MultiScaleRoIAlign, roi_align
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.rpn import RegionProposalNetwork, concat_box_prediction_layers

from .faster_rcnn_norm_aware import FasterRCNN_NormAware, CoordRegressor, NormAwareRoiHeads, NormAwareEmbeddingProj
from ..loss import OIMLossSMR
from .resnet_backbone import resnet_backbone


class FasterRCNN_NormAware_PW(FasterRCNN_NormAware):

    def _set_roi_heads(self, *args):
        return PixelWiseNormAwareRoiHeads(*args)

    def ex_feat_by_roi_pooling(self, images, targets):
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals = [x['boxes'] for x in targets]

        roi_pooled_features = self.roi_heads.box_roi_pool(
            features, proposals, images.image_sizes)
        rcnn_features = self.roi_heads.feat_head(roi_pooled_features)
        if isinstance(rcnn_features, torch.Tensor):
            rcnn_features = OrderedDict([('feat_res5', rcnn_features)])
        embeddings, class_logits = self.roi_heads.embedding_head(rcnn_features)
        spatial_attention = torch.sigmoid(class_logits)
        embeddings = F.adaptive_avg_pool2d(
            embeddings * spatial_attention, 1).flatten(start_dim=1)
        return embeddings.split(1, 0)

    def ex_feat_by_img_crop(self, images, targets):
        assert len(images) == 1, 'Only support batch_size 1 in this mode'

        images, targets = self.transform(images, targets)
        x1, y1, x2, y2 = map(lambda x: int(round(x)),
                             targets[0]['boxes'][0].tolist())
        input_tensor = images.tensors[:, :, y1:y2 + 1, x1:x2 + 1]
        features = self.backbone(input_tensor)
        features = features.values()[0]
        rcnn_features = self.roi_heads.feat_head(features)
        if isinstance(rcnn_features, torch.Tensor):
            rcnn_features = OrderedDict([('feat_res5', rcnn_features)])
        embeddings, class_logits = self.roi_heads.embedding_head(rcnn_features)
        spatial_attention = torch.sigmoid(class_logits)
        embeddings = F.adaptive_avg_pool2d(
            embeddings * spatial_attention, 1).flatten(start_dim=1)
        return embeddings.split(1, 0)


class PixelWiseNormAwareRoiHeads(NormAwareRoiHeads):

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                assert t["boxes"].dtype.is_floating_point, \
                    'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, \
                    'target labels must of int64 type'
                if self.has_keypoint:
                    assert t["keypoints"].dtype == torch.float32, \
                        'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets, matched_gt_boxes = \
                self.select_training_samples(proposals, targets)

        rcnn_features = self.feat_head(
            self.box_roi_pool(features, proposals, image_shapes)
        )  # size = (N, C, 7, 7)

        box_regression = self.box_predictor(rcnn_features['feat_res5'])
        embeddings_, class_logits = self.embedding_head(
            rcnn_features)  # size = (N, d, 7, 7) and (N, 1, 7, 7)
        spatial_attention = torch.sigmoid(class_logits)
        embeddings_ = F.adaptive_avg_pool2d(
            embeddings_ * spatial_attention, 1).flatten(start_dim=1)  # size = (N, d)

        result, losses = [], {}
        if self.training:
            # Generate pixel-wise label
            spatial_labels = self.grid_wise_label_gen(matched_gt_boxes, proposals,
                                                      size=rcnn_features['feat_res5'].shape[2:])
            det_labels = [y.clamp(0, 1) for y in labels]
            loss_detection, loss_box_reg = \
                spatial_norm_aware_rcnn_loss(class_logits, box_regression,
                                             spatial_labels, det_labels, regression_targets,
                                             self.focal, self.alpha_d, self.gamma_d)
            loss_reid = self.reid_loss(embeddings_, labels)

            losses = dict(loss_detection=loss_detection,
                          loss_box_reg=loss_box_reg,
                          loss_reid=loss_reid)
        else:
            boxes, scores, embeddings, labels = \
                self.postprocess_detections(class_logits, box_regression, embeddings_,
                                            proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                        embeddings=embeddings[i],
                    )
                )
        # Mask and Keypoint losses are deleted
        return result, losses

    def postprocess_detections(self, class_logits, box_regression, embeddings_, proposals, image_shapes):
        device = class_logits.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = torch.sigmoid(class_logits).mean(dim=(2, 3))
        pred_scores /= pred_scores.max()

        embeddings_ = F.normalize(embeddings_)
        embeddings_ = embeddings_ * pred_scores.view(-1, 1)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings_.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        for boxes, scores, embeddings, image_shape in zip(pred_boxes, pred_scores, pred_embeddings, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.ones(scores.size(0), device=device)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)
            # embeddings are already personized.

            # batch everything, by making every class prediction be a separate
            # instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            embeddings = embeddings.reshape(-1, self.embedding_head.dim)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, embeddings = boxes[
                inds], scores[inds], labels[inds], embeddings[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = boxes[keep], scores[keep], \
                labels[keep], embeddings[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels, embeddings = boxes[keep], scores[keep], \
                labels[keep], embeddings[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_embeddings.append(embeddings)

        return all_boxes, all_scores, all_embeddings, all_labels

    @staticmethod
    def grid_wise_label_gen(matched_gt_boxes, proposals, size=(7, 7)):
        cat_gt = torch.cat(matched_gt_boxes, dim=0)
        cat_p = torch.cat(proposals, dim=0)
        num_proposals = cat_gt.size(0)
        grid_wise_labels = torch.zeros(
            num_proposals, 1, *size).to(proposals[0].device)
        for i in range(num_proposals):
            width = (cat_p[i][2] - cat_p[i][0]).ceil().long().item()
            height = (cat_p[i][3] - cat_p[i][1]).ceil().long().item()
            if not (width > 0 and height > 0):
                continue  # invalid proposal
            tmp = torch.zeros(1, 1, height, width)  # same 'size' as proposal
            x1 = (torch.max(cat_p[i][0], cat_gt[i][0]) -
                  cat_p[i][0]).floor().long().item()
            y1 = (torch.max(cat_p[i][1], cat_gt[i][1]) -
                  cat_p[i][1]).floor().long().item()
            x2 = (torch.min(cat_p[i][2], cat_gt[i][2]) -
                  cat_p[i][0]).ceil().long().item()
            y2 = (torch.min(cat_p[i][3], cat_gt[i][3]) -
                  cat_p[i][1]).ceil().long().item()
            tmp[0, 0, y1:y2 + 1, x1:x2 + 1] = 1.0
            grid_wise_labels[i] = F.interpolate(
                tmp, size=size, mode='bilinear', align_corners=False)
        return grid_wise_labels

    def select_training_samples(self, proposals, targets):
        """
        https://github.com/pytorch/vision/blob/master/torchvision/models/detection/roi_heads.py#L445
        """
        self.check_targets(targets)
        dtype = proposals[0].dtype
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(
            proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            matched_gt_boxes.append(gt_boxes[img_id][matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets, matched_gt_boxes


class PixelWiseNormAwareEmbeddingProj(NormAwareEmbeddingProj):
    '''
    Current Version: 
        1. conv kernel shared by all locations
        2. average pool instead of resize/concat, so the output dim = embedding dim
    TODO: Version 2: don't share conv kernels.

    '''

    def __init__(self, *args, **kwargs):
        super(PixelWiseNormAwareEmbeddingProj, self).__init__(*args, **kwargs)
        self.rescaler = nn.BatchNorm2d(1, affine=True)
        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_chennel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            proj = nn.Sequential(
                nn.Conv2d(in_chennel, indv_dim, kernel_size=1),
                nn.BatchNorm2d(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

    def forward(self, featmaps):
        '''
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim, gird_size[0], grid_size[1]), L2 normalized embeddings.
            tensor of size (BatchSize, 1, gird_size[0], grid_size[1]) rescaled norm of embeddings, as class_logits.
        '''
        if len(featmaps) == 1:
            k, v = featmaps.items()[0]
            embeddings = self.projectors[k](v)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.clamp(min=1e-12)
            norms = self.rescaler(norms)
            return embeddings, norms
        else:
            outputs = []
            for k, v in featmaps.items():
                outputs.append(
                    self.projectors[k](v)
                )
            outputs[0] = F.interpolate(
                outputs[0], size=outputs[1].shape[2:],
                mode='bilinear', align_corners=False)
            embeddings = torch.cat(outputs, dim=1)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.clamp(min=1e-12)
            norms = self.rescaler(norms)
            return embeddings, norms

    @property
    def rescaler_weight(self):
        return self.rescaler.weight.item()


def spatial_norm_aware_rcnn_loss(class_logits, box_regression, spatial_labels, labels, regression_targets,
                                 focal=False, alpha_d=0.25, gamma_d=2.0):
    """
    Computes the loss for grid/pixel-wise Norm-Aware R-CNN.
    Arguments:
        class_logits (Tensor), size = (N, 1, h, w)
        box_regression (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.binary_cross_entropy_with_logits(
        class_logits, spatial_labels, reduction='none')
    if focal:
        pt = torch.exp(-classification_loss)
        classification_loss = alpha_d * (1 - pt)**gamma_d * classification_loss

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N = class_logits.size(0)
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss.mean(), box_loss


def get_pixel_wise_norm_aware_model(args, training=True, pretrained_backbone=True):
    phase_args = args.train if training else args.test

    if hasattr(args, 'embedding_feat_fuse') and args.embedding_feat_fuse:
        featmap_names = ['feat_res4', 'feat_res5']
        in_channels = [1024, 2048]
        return_res4 = True
    else:
        featmap_names = ['feat_res5']
        in_channels = [2048]
        return_res4 = False

    pool_module = MultiScaleRoIAlign
    net_fn = FasterRCNN_NormAware_PW

    if not hasattr(args.train, 'focal'):
        args.train.focal = False

    # --------------------------------------------------------------------#

    backbone, conv_head = resnet_backbone('resnet50', pretrained_backbone,
                                          return_res4=return_res4, GAP=False)

    roi_pooler = pool_module(
        featmap_names=['feat_res4'],
        output_size=14,
        sampling_ratio=2)

    coord_fc = CoordRegressor(2048, num_classes=2,
                              RCNN_bbox_bn=args.rcnn_bbox_bn)

    embedding_head = PixelWiseNormAwareEmbeddingProj(
        featmap_names=featmap_names,
        in_channels=in_channels,
        dim=256)

    if args.smr:
        reid_loss = OIMLossSMR(args.num_features, args.num_pids, args.num_cq_size,
                               args.train.oim_momentum, args.oim_scalar, args.smr_omega_decay)
    else:
        reid_loss = None  # fallback to the default OIM loss.

    model = net_fn(backbone,
                   feat_head=conv_head,
                   box_predictor=coord_fc,
                   embedding_head=embedding_head,
                   num_pids=args.num_pids, num_cq_size=args.num_cq_size,
                   min_size=phase_args.min_size, max_size=phase_args.max_size,
                   anchor_scales=(tuple(args.anchor_scales),),
                   anchor_ratios=(tuple(args.anchor_ratios),),
                   # RPN parameters
                   rpn_pre_nms_top_n_train=args.train.rpn_pre_nms_top_n,
                   rpn_post_nms_top_n_train=args.train.rpn_post_nms_top_n,
                   rpn_pre_nms_top_n_test=args.test.rpn_pre_nms_top_n,
                   rpn_post_nms_top_n_test=args.test.rpn_post_nms_top_n,
                   rpn_nms_thresh=phase_args.rpn_nms_thresh,
                   rpn_fg_iou_thresh=args.train.rpn_positive_overlap,
                   rpn_bg_iou_thresh=args.train.rpn_negative_overlap,
                   rpn_batch_size_per_image=args.train.rpn_batch_size,
                   rpn_positive_fraction=args.train.rpn_fg_fraction,
                   # Box parameters
                   box_roi_pool=roi_pooler,
                   rcnn_bbox_bn=args.rcnn_bbox_bn,
                   box_score_thresh=args.train.fg_thresh,
                   box_nms_thresh=args.test.nms,  # inference only
                   box_detections_per_img=phase_args.rpn_post_nms_top_n,  # use all
                   box_fg_iou_thresh=args.train.bg_thresh_hi,
                   box_bg_iou_thresh=args.train.bg_thresh_lo,
                   box_batch_size_per_image=args.train.rcnn_batch_size,
                   box_positive_fraction=args.train.fg_fraction,  # for proposals
                   bbox_reg_weights=args.train.box_regression_weights,
                   # reid loss
                   reid_loss=reid_loss
                   )
    model.roi_heads.focal = args.train.focal
    model.roi_heads.alpha_d = args.train.alpha_d
    model.roi_heads.gamma_d = args.train.gamma_d
    if training:
        model.train()
    else:
        model.eval()

    if hasattr(args, 'NAE_pretrain') and args.NAE_pretrain:
        model = load_NAE_weights(args, model)

    return model


def load_NAE_weights(args, model):
    if args.dataset == 'CUHK-SYSU':
        model_path = 'logs/cuhk-sysu/checkpoint.pth'
    elif args.dataset == 'PRW':
        model_path = 'logs/prw/checkpoint.pth'
    checkpoint = torch.load(model_path)

    state_dict = checkpoint['model']
    state_dict['roi_heads.embedding_head.projectors.feat_res4.0.weight'] = \
        state_dict['roi_heads.embedding_head.projectors.feat_res4.0.weight'].view(
            128, 1024, 1, 1)
    state_dict['roi_heads.embedding_head.projectors.feat_res5.0.weight'] = \
        state_dict['roi_heads.embedding_head.projectors.feat_res5.0.weight'].view(
            128, 2048, 1, 1)

    model.load_state_dict(state_dict)
    print(hue.good('NAE pre-trained weights loaded.'))
    return model
