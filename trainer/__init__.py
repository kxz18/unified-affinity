#!/usr/bin/python
# -*- coding:utf-8 -*-
import utils.register as R

from .affinity_trainer import AffinityTrainer


def create_trainer(config, model, train_loader, valid_loader):
    return R.construct(
        config['trainer'],
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        save_config=config)


