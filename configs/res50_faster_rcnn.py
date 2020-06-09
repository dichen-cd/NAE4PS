import argparse
import torchvision as tv


def args_faster_rcnn():
    parser = argparse.ArgumentParser(
        add_help=False,
        description='Plain Faster R-CNN')

    parser.add_argument('-p', '--path', dest='path',
                        help='directory to save models', default='logs/')
    parser.add_argument('--debug', action='store_true')
    # Data
    parser.add_argument('--dataset',
                        help='training dataset',
                        default='CUHK-SYSU', type=str,
                        choices=['CUHK-SYSU', 'PRW'])
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    # Net architecture
    parser.add_argument('--net',
                        default='resnet50', type=str,
                        choices=tv.models.resnet.__all__)
    parser.add_argument('--rm_rcnn_bbox_bn', dest='rcnn_bbox_bn',
                        help='whether to use batch normalization for dc_rcc_box_regression',
                        action='store_false')
    parser.add_argument('--anchor_scales', type=float, nargs='+', default=(32, 64, 128, 256, 512),
                        help='ANCHOR_SCALES w.r.t. image size.')
    parser.add_argument('--anchor_ratios', type=float, nargs='+', default=(0.5, 1., 2.),
                        help='ANCHOR_RATIOS: anchor height/width')
    # resume trained model
    parser.add_argument('--resume',
                        help='resume file path',
                        default=None, type=str)
    # Device
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--apex', dest='apex',
                        help='Whether to use apex mixed precision / distributed training.',
                        action='store_true')
    parser.add_argument('--distributed', dest='distributed',
                        help='Whether to use distributed training.',
                        action='store_true')
    parser.add_argument('--world_size', type=int,
                        help='Number of distributed processes. (optional)')
    parser.add_argument('--rank', type=int, help='Rank of process. (optional)')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Random Seed
    parser.add_argument('--seed', type=int, default=1)

    #
    # Training
    #
    parser.add_argument('--wo_pretrained', dest='train.wo_pretrained',
                        help='whether to disable ImageNet pretrained weights.',
                        action='store_true')
    parser.add_argument('--start_epoch', dest='train.start_epoch',
                        help='starting epoch',
                        default=0, type=int)
    parser.add_argument('--epochs', dest='train.epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='train.disp_interval',
                        help='number of iterations to display',
                        default=10, type=int)
    # Training.Optimization
    parser.add_argument('--lr', dest='train.lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--momentum', dest='train.momentum',
                        help='Momentum',
                        default=0.9, type=float)
    parser.add_argument('--weight_decay', dest='train.weight_decay',
                        help='Weight Decay',
                        default=0.0005, type=float)
    parser.add_argument('--lr_decay_gamma', dest='train.lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    # Training.Optimization.lr_decay
    parser.add_argument('--lr_decay_step', dest='train.lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=16, type=int)
    parser.add_argument('--lr_decay_milestones', type=int, dest='train.lr_decay_milestones',
                        nargs='+', default=None)
    parser.add_argument('--lr_warm_up', dest='train.lr_warm_up',
                        action='store_true')
    parser.add_argument('--clip_gradient', dest='train.clip_gradient',
                        type=float, default=10.0)
    # Training.tricks
    parser.add_argument('--double_bias', dest='train.double_bias',
                        type=int, default=0, choices=[0, 1],
                        help='Whether to double the learning rate for bias')
    parser.add_argument('--truncated', dest='train.truncated',
                        action='store_true',
                        help='Whether to initialize the weights with truncated normal distribution')
    parser.add_argument('--bias_decay', dest='train.bias_decay',
                        action='store_true',
                        help='Whether to have weight decay on bias as well')
    # Training.data
    parser.add_argument('--aspect_grouping', dest='train.aspect_grouping',
                        type=int, default=-1,
                        help='Whether to use aspect-ratio grouping of training images, \
                              introduced merely for saving GPU memory')
    parser.add_argument('--min_size', dest='train.min_size',
                        type=int, default=900,
                        help='Minimum size of the image to be rescaled before feeding \
                              it to the backbone')
    parser.add_argument('--max_size', dest='train.max_size',
                        type=int, default=1500,
                        help='Max pixel size of the longest side of a scaled input image')
    parser.add_argument('--batch_size', dest='train.batch_size',
                        default=5, type=int,
                        help='batch_size, __C.TRAIN.IMS_PER_BATCH')
    parser.add_argument('--no_flip', dest='train.use_flipped',
                        action='store_false',
                        help='Use horizontally-flipped images during training?')
    # Training.data.rcnn/rpn.sampling
    parser.add_argument('--rcnn_batch_size', dest='train.rcnn_batch_size',
                        type=int, default=128,
                        help='Minibatch size (number of regions of interest [ROIs])\
                              __C.TRAIN.BATCH_SIZE')
    parser.add_argument('--fg_fraction', dest='train.fg_fraction',
                        type=float, default=0.5,
                        help='Fraction of minibatch that is labeled foreground (i.e. class > 0)')
    parser.add_argument('--fg_thresh', dest='train.fg_thresh',
                        type=float, default=0.5,
                        help='Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)')
    parser.add_argument('--bg_thresh_hi', dest='train.bg_thresh_hi',
                        type=float, default=0.5,
                        help='Overlap threshold for a ROI to be considered background')
    parser.add_argument('--bg_thresh_lo', dest='train.bg_thresh_lo',
                        type=float, default=0.1,
                        help='Overlap threshold for a ROI to be considered background')
    parser.add_argument('--box_regression_weights', dest='train.box_regression_weights',
                        type=float, nargs=4, default=[10., 10., 5., 5.],
                        help='Weights for the encoding/decoding of the bounding boxes')

    # parser.add_argument('--bbox_thresh', dest='train.bbox_thresh',
    #                     type=float, default=0.5,
    #                     help='Overlap required between a ROI and ground-truth box in order for that ROI to\
    # be used as a bounding-box regression training example')

    # Training.RPN
    parser.add_argument('--rpn_positive_overlap', dest='train.rpn_positive_overlap',
                        type=float, default=0.7,
                        help='IOU >= thresh: positive example')
    parser.add_argument('--rpn_negative_overlap', dest='train.rpn_negative_overlap',
                        type=float, default=0.3,
                        help='IOU < thresh: negative example')
    parser.add_argument('--rpn_fg_fraction', dest='train.rpn_fg_fraction',
                        type=float, default=0.5,
                        help='Max ratio of foreground examples.')
    parser.add_argument('--rpn_batch_size', dest='train.rpn_batch_size',
                        type=int, default=256,
                        help='Total number of examples')
    parser.add_argument('--rpn_nms_thresh', dest='train.rpn_nms_thresh',
                        type=float, default=0.7,
                        help='NMS threshold used on RPN proposals')
    parser.add_argument('--rpn_pre_nms_top_n', dest='train.rpn_pre_nms_top_n',
                        type=int, default=12000,
                        help='Number of top scoring boxes to keep before apply NMS to RPN proposals')
    parser.add_argument('--rpn_post_nms_top_n', dest='train.rpn_post_nms_top_n',
                        type=int, default=2000,
                        help='Number of top scoring boxes to keep before apply NMS to RPN proposals')
    parser.add_argument('--rpn_min_size', dest='train.rpn_min_size',
                        type=int, default=8,
                        help='Proposal height and width both need to be greater than RPN_MIN_SIZE (at\
                              orig image scale)')
    # Training.checkpointing
    parser.add_argument('--checkpoint_interval', dest='train.checkpoint_interval',
                        type=int, default=1,
                        help='Epochs between snapshots.')
    # Training. log and diaplay
    parser.add_argument('--no_tfboard', dest='train.use_tfboard',
                        help='whether use tensorflow tensorboard',
                        action='store_false')
    # Training.loss weights
    parser.add_argument('--w_RPN_loss_cls', dest='train.w_RPN_loss_cls',
                        default=1.0, type=float)
    parser.add_argument('--w_RPN_loss_box', dest='train.w_RPN_loss_box',
                        default=1.0, type=float)
    parser.add_argument('--w_RCNN_loss_bbox', dest='train.w_RCNN_loss_bbox',
                        default=1.0, type=float)
    parser.add_argument('--w_RCNN_loss_cls', dest='train.w_RCNN_loss_cls',
                        default=1.0, type=float)

    #
    # Test
    #
    parser.add_argument('--min_size_test', dest='test.min_size',
                        type=int, default=900,
                        help='Minimum size of the image to be rescaled before feeding \
                              it to the backbone')
    parser.add_argument('--max_size_test', dest='test.max_size',
                        type=int, default=1500,
                        help='Max pixel size of the longest side of a scaled input image')
    parser.add_argument('--batch_size_test', dest='test.batch_size',
                        default=1, type=int,
                        help='batch_size')
    parser.add_argument('--nms_test', dest='test.nms',
                        type=float, default=0.4,
                        help='NMS threshold used on RCNN output')
    parser.add_argument('--rpn_nms_thresh_test', dest='test.rpn_nms_thresh',
                        type=float, default=0.7,
                        help='NMS threshold used on RPN proposals')
    parser.add_argument('--rpn_pre_nms_top_n_test', dest='test.rpn_pre_nms_top_n',
                        type=int, default=6000,
                        help='Number of top scoring boxes to keep before apply NMS to RPN proposals')
    parser.add_argument('--rpn_post_nms_top_n_test', dest='test.rpn_post_nms_top_n',
                        type=int, default=300,
                        help='Number of top scoring boxes to keep before apply NMS to RPN proposals')
    parser.add_argument('--rpn_min_size_test', dest='test.rpn_min_size',
                        type=int, default=16,
                        help='Proposal height and width both need to be greater than RPN_MIN_SIZE (at\
                              orig image scale)')

    return parser


def args_faster_rcnn_oim(meta_parser=args_faster_rcnn):
    parser = argparse.ArgumentParser(
        add_help=False,
        parents=[meta_parser()],
        description='OIM model based on Faster R-CNN')
    parser.add_argument('--smr', action='store_true',
                        help='Weather to use OIMLossSMR instead of OIMLoss')
    parser.add_argument('--smr_omega_decay', type=float, default=0.99,
                        help='Decay factor of the importance weights of SMR')
    parser.add_argument('--embedding_feat_fuse', action='store_true',
                        help='Weather to fuse feat_res4 and feat_res5')

    # sizes
    parser.add_argument('--num_features', type=int, default=256,
                        help='Embedding dimension.')
    parser.add_argument('--num_pids', type=int, default=5532,
                        choices=[5532, 482],
                        help='Labeled person ids in each dataset.')
    parser.add_argument('--num_cq_size', type=int, default=5000,
                        help='Size of circular queue for unlabeled persons')
    parser.add_argument('--oim_scalar', type=float, default=30.0,
                        help='1 / OIM temperature')

    # training
    parser.add_argument('--w_OIM_loss_oim', dest='train.w_OIM_loss_oim',
                        default=1.0, type=float)
    parser.add_argument('--oim_momentum', dest='train.oim_momentum',
                        default=0.5, type=float)
    # training.focal_loss.
    # r: reid, d:detection
    parser.add_argument('--focal', dest='train.focal', action='store_true',
                        help='Whether to use focal loss on detection and reid losses.')
    parser.add_argument('--alpha_d', dest='train.alpha_d',
                        default=0.25, type=float)
    parser.add_argument('--alpha_r', dest='train.alpha_r',
                        default=0.25, type=float)
    parser.add_argument('--gamma_d', dest='train.gamma_d',
                        default=2.0, type=float)
    parser.add_argument('--gamma_r', dest='train.gamma_r',
                        default=2.0, type=float)

    return parser


def args_faster_rcnn_norm_aware(meta_parser=args_faster_rcnn_oim):
    parser = argparse.ArgumentParser(
        add_help=False,
        parents=[meta_parser()],
        description='Norm-Aware model based on Faster R-CNN')
    parser.add_argument('--pixel_wise', dest='pixel_wise',
                        action='store_true',
                        help='Wether to use pixel-wise norm-aware model.')
    parser.add_argument('--NAE_pretrain', dest='NAE_pretrain',
                        action='store_true',
                        help='Whether to use NAE weights for NAE+. ')

    return parser
