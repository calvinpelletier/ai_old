#!/usr/bin/env python3
import os
import time
import json
import pickle
import torch
from external.op import conv2d_gradfix
from external.op import grid_sample_gradfix
from ai_old.util.factory import build_trainer, build_dataset, build_task
import copy
from ai_old.util.etc import make_deterministic
from torch.utils.tensorboard import SummaryWriter


def training_loop(cfg):
    start_time = time.time()

    make_deterministic(seed=cfg.seed)

    batch_size = cfg.dataset.batch_size
    latest_path = os.path.join(cfg.saves_dir, 'latest.pkl')

    device = torch.device('cuda', 0)

    torch.backends.cudnn.benchmark = True
    conv2d_gradfix.enabled = True
    grid_sample_gradfix.enabled = True

    # init logs
    print('[INFO] initializing logs...')
    logger = SummaryWriter(log_dir=cfg.logs_dir)

    # dataset
    print('[INFO] building datasets...')
    training_set_iterator = None
    if cfg.dataset.type is not None:
        dataset_core = build_dataset(cfg.dataset)
        if dataset_core.has_training_set():
            training_set_iterator = dataset_core.get_train_set(
                batch_size,
                cfg.seed,
                rank,
                num_gpus,
            )

    # trainer
    trainer = build_trainer(cfg, logger, device)

    # task
    task = build_task(cfg, logger, device)

    # progress tracking
    cur_step = 0
    tick_start_step = cur_step
    tick_start_time = time.time()
    batch_idx = 0

    assert not cfg.resume, 'todo'

    # print info
    print('~~~~~config~~~~~')
    print(json.dumps(cfg, indent=2))
    print('~~~~~~~~~~~~~~~~')
    trainer.print_models()

    # train
    print(f'[INFO] training for {cfg.n_steps} steps...')
    while True:
        # check conditions
        is_final_round = (cur_step >= cfg.n_steps)
        do_eval = is_final_round or (cur_step % cfg.eval_freq) < batch_size
        do_logging = do_eval or \
            (cur_step % cfg.logging_freq) < batch_size

        if do_logging:
            # print info
            tick_end_time = time.time()
            msg = 'step={}, steps/s={:.2f}'.format(
                cur_step,
                (cur_step - tick_start_step) / (tick_end_time - tick_start_time),
            )
            pad = 100 - len(msg)
            halfpad = pad // 2
            print('~' * halfpad + msg + '~' * (pad - halfpad))

            # update state
            tick_start_step = cur_step
            tick_start_time = time.time()

        # fetch data
        if training_set_iterator is not None:
            with torch.autograd.profiler.record_function('data_fetch'):
                batch = next(training_set_iterator)
        else:
            batch = None

        # run batch through trainer
        trainer.run_batch(batch, batch_idx, cur_step, do_logging)

        # update state
        cur_step += batch_size
        batch_idx += 1

        if do_eval:
            # save
            print('[INFO] saving model...')
            snapshot_data = {}
            for name, module in trainer.get_modules_for_save():
                if module is not None:
                    module = copy.deepcopy(
                        module,
                    ).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            resume_path = os.path.join(cfg.saves_dir, 'latest.pkl')
            with open(resume_path, 'wb') as f:
                pickle.dump(snapshot_data, f)

            with torch.no_grad():
                # eval
                task.eval(trainer, snapshot_data, cur_step)

                # sample
                task.sample(trainer, snapshot_data, cur_step)

            del snapshot_data # conserve memory

        if done:
            break

    # Done.
    print('exiting...')
