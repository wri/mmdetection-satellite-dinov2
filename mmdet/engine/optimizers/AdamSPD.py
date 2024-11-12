from mmdet.registry import OPTIMIZERS
from torch.optim import Optimizer
import torch
#from torch.optim.optimizer import Optimizer, required
import copy
import math
import pickle

@OPTIMIZERS.register_module()
class AdamSPD(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
       # params_to_opt = [x[0] for x in module.named_parameters() if x[1].requires_grad and not x[0].startswith("fpn")]
        #print(params_to_opt)
        #with open('/home/ubuntu/params_to_opt.pkl', 'rb') as handle:
        #     params_to_opt = pickle.load(handle)
        #with open('/home/ubuntu/params_name.pkl', 'rb') as handle:
        #     params_name = pickle.load(handle)
        #params_anchor = copy.deepcopy(params_to_opt) # Cache pre-trained model weights 
        #params = [{'params':params_to_opt,
        #        'pre': params_anchor,
        #         'name': params_name}]
        #param_groups = params
        super(AdamSPD, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(AdamSPD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            #print(group['weight_decay'])
            #print(group['lr'])
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['hyper'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        
                       

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            self.adam(group,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   state_steps,
                   group['amsgrad'],
                   beta1,
                   beta2,
                   group['lr'],
                   group['weight_decay'],
                   group['eps'],
                   group['name']
                   )
        return loss

    def adam(self, group,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps, 
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float,
         name):
        
        i = 0
        #print(group['params'])
        #print(group['weight_decay'])
        for j, param in enumerate(group['params']):
            if param.grad is None: 
                continue
            grad = param.grad
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]
            if amsgrad:
                max_exp_avg_sq = max_exp_avg_sqs[i]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            
        
            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            
            step_size = lr / bias_correction1
            
            
            d_p = step_size * exp_avg/denom 
            new_p = param - d_p

            # Selective Projection Decay (SPD)
            #print(weight_decay, group['params'][j])
            pre = group['pre'][j] if group['pre'] is not None else torch.zeros_like(param)
            
            if 'backbone' not in name or 'fpn' in name:
                new_p = new_p -  weight_decay * new_p
                condition = 0
            else:
                condition = - torch.sum(torch.mul(grad, param - pre))
                
            if condition < 0.0:
                ratio = self._ratio(new_p, param, pre)
                new_p = new_p - weight_decay * ratio * (new_p - pre)
            param.copy_(new_p)
            i += 1

    def _ratio(self,new_p,param,pre):
        curr_norm, prev_norm = torch.norm(new_p - pre), torch.norm(param - pre)
        ratio = (curr_norm - prev_norm) / curr_norm 
        return torch.nn.functional.hardtanh(ratio, 0.0, 1.0)
