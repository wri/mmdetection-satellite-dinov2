import json
from mmengine.dist import get_dist_info
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS
import copy
import torch


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LayerDecayOptimizerConstructor_ViT(DefaultOptimWrapperConstructor):
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.
        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """

        parameter_groups = {}

        #params_anchor = copy.deepcopy(params_to_opt)
        for i, value in enumerate(module.named_parameters()):
            if not value[1].requires_grad:
                continue  # frozen weights
            group_name = value[0]

            this_weight_decay = 0.05 if ('backbone' in value[0]) else 1e-4
            scale = 1e-1 if (('backbone' in value[0] and 'fpn' not in value[0])) else 1
            this_weight_decay = this_weight_decay if 'norm' not in value[0] else 0
            this_weight_decay = this_weight_decay if 'cls_token' not in value[0] else 0
            this_weight_decay = this_weight_decay if 'dist_token' not in value[0] else 0
            this_weight_decay = this_weight_decay if 'mask_token' not in value[0] else 0
            this_weight_decay = this_weight_decay if 'pos_embed' not in value[0] else 0
            
            parameter_groups[group_name] = {
                        'weight_decay': this_weight_decay,
                        'params': value[1],
                        'param_names': [],
                        'pre': [],
                        'lr_scale': scale,
                        'name': group_name,
                        'lr': scale * self.base_lr,
                    }
            if 'backbone' in value[0]:
                parameter_groups[group_name]['pre'].append(copy.deepcopy(value[1]))
            else:
                parameter_groups[group_name]['pre'].append(torch.tensor([0], device = torch.device('cuda')))
        params.extend(parameter_groups.values())

