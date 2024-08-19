from detrex.config import get_config
from .models.stabledino_vit import model
from detrex.config.configs.common.common_schedule import multistep_lr_scheduler


# get default config
dataloader = get_config("common/data/coco_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
#lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep_warmup

# define `lr_multiplier` config
lr_multiplier = multistep_lr_scheduler(
    values=[1.0, 0.1], 
    warmup_steps=250, 
    num_updates=50000, 
    milestones=[45000], 
    warmup_method="linear", 
    warmup_factor=0.001, 
)
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = (
   "/home/ubuntu/mmdetection/models/detectronssl.pkl"
)
train.output_dir = "./output/dino_r50_4scale_12ep"

# max training iterations
train.max_iter = 50000

# run evaluation every 5000 iters
train.eval_period = 1000

# log training infomation every 20 iters
train.log_period = 20

# set random seed
train.seed = 42

# save checkpoint every 5000 iters
train.checkpointer.period = 3000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

train.test_with_nms=0.1

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4

#def my_factor_func 
optimizer.params.lr_factor_func = lambda module_name: 0.1 if (('backbone' in module_name) or ('dist_token' in module_name) or ('pos_embed' in module_name)) and 'fpn' not in module_name else 1 #any(module_name in ["backbone.blocks", "backbone.norm", "backbone.pos_embed", "backbone.dist_token", "backbone.patch_embed"]) else 1
optimizer.params.overrides = {#'backbone.backbone': {'weight_decay': 0.01},
                                    #'backbone.norm': {'lr_mult':0.1, 'lr_factor' : 0.1, 'weight_decay' : 0.01},
                                    #'backbone.backbone.blocks': {'weight_decay' : 0.01},
                                    'pos_embed': {'lr_mult':0.1, 'lr_factor' : 0.1, 'wd_mult' : 0, 'weight_decay' : 0},
                                    'dist_token': {'lr_mult':0.1, 'lr_factor' : 0.1, 'wd_mult' : 0, 'weight_decay' : 0},
                                    'cls_token': {'lr' : 0, 'lr_factor' : 0., 'wd_mult' : 0, 'weight_decay' : 0},
                                    'mask_token': {'lr':0.0},
                                    #'backbone.patch_embed': {'weight_decay' : 0.01},
                                   }
# modify dataloader config
dataloader.train.num_workers = 2

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 2

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
