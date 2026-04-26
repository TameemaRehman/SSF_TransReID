from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def log_device_info(logger):
    """Log whether training is on CUDA or CPU (file + console via logger)."""
    logger.info("")
    logger.info("=" * 60)
    if torch.cuda.is_available():
        logger.info(f"  ✅ Training on GPU : {torch.cuda.get_device_name(0)}")
        logger.info(
            f"  📦 GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
        logger.info(f"  🔧 CUDA Version    : {torch.version.cuda}")
    else:
        logger.info("  ⚠️  Training on CPU  (no CUDA GPU detected)")
    logger.info("=" * 60)

def log_trainable_params(logger, model):
    """Log total, trainable, and frozen parameter counts to file and console."""
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params

    logger.info("")
    logger.info("=" * 60)
    logger.info("  📊 Model Parameter Summary")
    logger.info("-" * 60)
    logger.info(f"  Total parameters      : {total_params:>12,}")
    logger.info(f"  Trainable parameters  : {trainable_params:>12,}  ✅")
    logger.info(f"  Frozen parameters     : {frozen_params:>12,}  🔒")
    logger.info(f"  Trainable ratio       : {100 * trainable_params / total_params:>11.4f}%")
    logger.info("=" * 60)

    logger.info("  🔍 Trainable parameter groups:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"     {name:60s} | {param.numel():>10,} params")
    logger.info("=" * 60)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # ── Device info (stdout + train_log.txt) ───────────────────
    log_device_info(logger)

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    # ── Parameter summary (stdout + train_log.txt) ─────────────
    log_trainable_params(logger, model)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, args.local_rank
    )