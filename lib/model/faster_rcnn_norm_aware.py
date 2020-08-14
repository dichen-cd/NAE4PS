from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from .resnet_backbone import resnet_backbone

from ..loss import OIMLoss


class FasterRCNN_NormAware(GeneralizedRCNN):
    """
    See https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py#L26
    """

    def __init__(self, backbone,
                 num_classes=None, num_pids=5532, num_cq_size=5000,
                 # transform parameters
                 min_size=900, max_size=1500,
                 image_mean=None, image_std=None,
                 # Anchor settings:
                 anchor_scales=None, anchor_ratios=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=12000, rpn_pre_nms_top_n_test=6000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=300,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 rcnn_bbox_bn=True,
                 box_roi_pool=None, feat_head=None, box_predictor=None,
                 box_score_thresh=0.0, box_nms_thresh=0.4, box_detections_per_img=300,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.1,
                 box_batch_size_per_image=128, box_positive_fraction=0.5,
                 bbox_reg_weights=None,
                 # ReID parameters
                 embedding_head=None, reid_loss=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                'backbone should contain an attribute out_channels '
                'specifying the number of output channels (assumed to be the '
                'same for all the levels)')

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError(
                    'num_classes should be None when box_predictor is specified')
        else:
            if box_predictor is None:
                raise ValueError('num_classes should not be None when box_predictor'
                                 'is not specified')

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            if anchor_scales is None:
                anchor_scales = ((32, 64, 128, 256, 512),)
            if anchor_ratios is None:
                anchor_ratios = ((0.5, 1.0, 2.0),)
            rpn_anchor_generator = AnchorGenerator(
                anchor_scales, anchor_ratios)

        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels,
                rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = self._set_rpn(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['feat_res4'],
                output_size=14,
                sampling_ratio=2)

        if feat_head is None:
            raise ValueError('feat_head should be specified manually.')

        if box_predictor is None:
            box_predictor = CoordRegressor(
                2048, num_classes,
                rcnn_bbox_bn)

        if embedding_head is None:
            embedding_head = NormAwareEmbeddingProj(
                featmap_names=['feat_res4', 'feat_res5'],
                in_channels=[1024, 2048],
                dim=256)

        if reid_loss is None:
            reid_loss = OIMLoss(
                256, num_pids, num_cq_size,
                0.5, 30.0)

        roi_heads = self._set_roi_heads(
            embedding_head, reid_loss,
            box_roi_pool, feat_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(
            min_size, max_size, image_mean, image_std)

        super(FasterRCNN_NormAware, self).__init__(
            backbone, rpn, roi_heads, transform)

    def _set_rpn(self, *args):
        return RegionProposalNetwork(*args)

    def _set_roi_heads(self, *args):
        return NormAwareRoiHeads(*args)

    def ex_feat(self, images, targets, mode='det'):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result: (tuple(Tensor)): list of 1 x d embedding for the RoI of each image

        """
        if mode == 'det':
            return self.ex_feat_by_roi_pooling(images, targets)
        elif mode == 'reid':
            return self.ex_feat_by_img_crop(images, targets)

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
        embeddings, norms = self.roi_heads.embedding_head(rcnn_features)
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
        embeddings, norms = self.roi_heads.embedding_head(rcnn_features)
        return embeddings.split(1, 0)


class NormAwareRoiHeads(RoIHeads):

    def __init__(self, embedding_head, reid_loss, *args, **kwargs):
        super(NormAwareRoiHeads, self).__init__(*args, **kwargs)
        self.embedding_head = embedding_head
        self.reid_loss = reid_loss

    @property
    def feat_head(self):  # re-name
        return self.box_head

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
            proposals, matched_idxs, labels, regression_targets = \
                self.select_training_samples(proposals, targets)

        roi_pooled_features = \
            self.box_roi_pool(features, proposals, image_shapes)
        rcnn_features = self.feat_head(roi_pooled_features)
        box_regression = self.box_predictor(rcnn_features['feat_res5'])
        embeddings_, class_logits = self.embedding_head(rcnn_features)

        result, losses = [], {}
        if self.training:
            det_labels = [y.clamp(0, 1) for y in labels]
            loss_detection, loss_box_reg = \
                norm_aware_rcnn_loss(class_logits, box_regression,
                                     det_labels, regression_targets)

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

        pred_scores = torch.sigmoid(class_logits)
        embeddings_ = embeddings_ * pred_scores.view(-1, 1)  # CWS

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


class NormAwareEmbeddingProj(nn.Module):

    def __init__(self, featmap_names=['feat_res5'],
                 in_channels=[2048],
                 dim=256):
        super(NormAwareEmbeddingProj, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = map(int, in_channels)
        self.dim = int(dim)

        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_chennel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            proj = nn.Sequential(
                nn.Linear(in_chennel, indv_dim),
                nn.BatchNorm1d(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

        self.rescaler = nn.BatchNorm1d(1, affine=True)

    def forward(self, featmaps):
        '''
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        '''
        if len(featmaps) == 1:
            k, v = featmaps.items()[0]
            v = self._flatten_fc_input(v)
            embeddings = self.projectors[k](v)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / \
                norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms
        else:
            outputs = []
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                outputs.append(
                    self.projectors[k](v)
                )
            embeddings = torch.cat(outputs, dim=1)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / \
                norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms

    @property
    def rescaler_weight(self):
        return self.rescaler.weight.item()

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x  # ndim = 2, (N, d)

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim / parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp


class CoordRegressor(nn.Module):
    """
    bounding box regression layers, without classification layer.
    for Fast R-CNN.
    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
                           default = 2 for pedestrian detection
    """

    def __init__(self, in_channels, num_classes=2, RCNN_bbox_bn=True):
        super(CoordRegressor, self).__init__()
        if RCNN_bbox_bn:
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels, 4 * num_classes),
                nn.BatchNorm1d(4 * num_classes))
            init.normal_(self.bbox_pred[0].weight, std=0.01)
            init.normal_(self.bbox_pred[1].weight, std=0.01)
            init.constant_(self.bbox_pred[0].bias, 0)
            init.constant_(self.bbox_pred[1].bias, 0)
        else:
            self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
            init.normal_(self.bbox_pred.weight, std=0.01)
            init.constant_(self.bbox_pred.bias, 0)
        self.cls_score = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        if x.ndimension() == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas


def norm_aware_rcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Norm-Aware R-CNN.
    Arguments:
        class_logits (Tensor), size = (N, )
        box_regression (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.binary_cross_entropy_with_logits(
        class_logits, labels.float())

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

    return classification_loss, box_loss


def get_norm_aware_model(args, training=True, pretrained_backbone=True):
    backbone, conv_head = resnet_backbone('resnet50', pretrained_backbone)
    coord_fc = CoordRegressor(2048, num_classes=2,
                              RCNN_bbox_bn=args.rcnn_bbox_bn)
    embedding_head = NormAwareEmbeddingProj(
        featmap_names=['feat_res4', 'feat_res5'],
        in_channels=[1024, 2048],
        dim=256)
    phase_args = args.train if training else args.test
    model = FasterRCNN_NormAware(backbone,
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
                                 rcnn_bbox_bn=args.rcnn_bbox_bn,
                                 box_score_thresh=args.train.fg_thresh,
                                 box_nms_thresh=args.test.nms,  # inference only
                                 box_detections_per_img=phase_args.rpn_post_nms_top_n,  # use all
                                 box_fg_iou_thresh=args.train.bg_thresh_hi,
                                 box_bg_iou_thresh=args.train.bg_thresh_lo,
                                 box_batch_size_per_image=args.train.rcnn_batch_size,
                                 box_positive_fraction=args.train.fg_fraction,  # for proposals
                                 bbox_reg_weights=args.train.box_regression_weights
                                 )
    if training:
        model.train()
    else:
        model.eval()

    return model
