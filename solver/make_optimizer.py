import torch


def make_optimizer(cfg, model, center_criterion):
    params = []
    ssf_lr = cfg.PEFT.SSF.LR if cfg.PEFT.SSF.LR > 0 else cfg.SOLVER.BASE_LR * 10

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        if 'ssf' in key:
            # SSF parameters: higher LR (10x default), zero weight decay.
            # Zero weight decay is important — regularising scale/shift
            # parameters toward zero would counteract their purpose.
            lr = ssf_lr
            weight_decay = 0.0
        elif "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        # FIX: do NOT pass global lr or weight_decay here.
        # Passing them overrides the per-parameter settings built above,
        # which means SSF's custom lr and zero weight_decay get ignored.
        # Passing params alone correctly uses each param group's own lr/wd.
        optimizer = torch.optim.AdamW(params)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)

    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center