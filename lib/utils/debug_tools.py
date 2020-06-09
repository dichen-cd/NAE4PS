import torch

from .distributed import reduce_dict


def get_rcnn_fg_bg_ratio(args, model):

    def _register_debug_hook(engine):

        def _get_rcnn_fg_bg_ratio(module, input, output):
            if engine.state.iteration % args.train.disp_interval == 0:
                targets = torch.cat(input[1])
                num_fg = targets.gt(0).sum()
                num_bg = targets.eq(0).sum()
                debug_info = {'num_fg': num_fg,
                              'num_bg': num_bg}
                debug_info_reduced = reduce_dict(debug_info, average=False)
                engine.state.debug_info = {k: v.item()
                                           for k, v in debug_info_reduced.items()}

        if hasattr(model, 'module'):
            model.module.roi_heads.reid_loss.register_forward_hook(_get_rcnn_fg_bg_ratio)
        else:
            model.roi_heads.reid_loss.register_forward_hook(_get_rcnn_fg_bg_ratio)

    return _register_debug_hook
