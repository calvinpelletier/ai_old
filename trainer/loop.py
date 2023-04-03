#!/usr/bin/env python3
import os
import time
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import external.sg2.misc as misc
from external.sg2.util import construct_class_by_name, open_url, format_time
from external.op import conv2d_gradfix
from external.op import grid_sample_gradfix
import external.sg2.legacy as legacy
from ai_old.util.etc import AttrDict
from external.sg2 import training_stats
from external.sg2.dataset import ImageFolderDataset
from ai_old.util.factory import build_trainer, build_dataset, build_task
from ai_old.util.etc import print_
import copy

# TODO: remove
from ai_old.util.factory import build_model_from_exp
from ai_old.util.etc import check_model_equality


def training_loop(rank, cfg):
    start_time = time.time()

    batch_size = cfg.dataset.batch_size
    batch_gpu = cfg.dataset.batch_gpu
    num_gpus = cfg.num_gpus
    latest_path = os.path.join(cfg.saves_dir, 'latest.pkl')
    metrics = ['fid50k_full']

    device = torch.device('cuda', rank)

    np.random.seed(cfg.seed * num_gpus + rank)
    torch.manual_seed(cfg.seed * num_gpus + rank)

    torch.backends.cudnn.benchmark = True
    conv2d_gradfix.enabled = True
    grid_sample_gradfix.enabled = True

    # dataset
    print_(rank, '[INFO] building datasets...')
    # training_set = ImageFolderDataset(
    #     path='/home/asiu/datasets/ffhq-128/x',
    #     resolution=128,
    #     max_size=None,
    #     use_labels=False,
    #     xflip=True,
    #     random_seed=cfg.seed,
    # )
    # training_set_sampler = misc.InfiniteSampler(
    #     dataset=training_set,
    #     rank=rank,
    #     num_replicas=num_gpus,
    #     seed=cfg.seed,
    # )
    # training_set_iterator = iter(torch.utils.data.DataLoader(
    #     dataset=training_set,
    #     sampler=training_set_sampler,
    #     batch_size=batch_size // num_gpus,
    #     pin_memory=True,
    #     num_workers=3,
    #     prefetch_factor=2,
    # ))
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
    trainer = build_trainer(cfg, rank, device)
    # if load_latest:
    #     self.total_steps = self.trainer.load_latest()

    # task
    task = build_task(cfg, rank, device)

    # init logs
    print_(rank, '[INFO] initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')

    # progress tracking
    cur_step = 0
    tick_start_step = cur_step
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0

    if cfg.resume:
        # TODO
        cur_step = 800000

    # print info
    if rank == 0:
        print('~~~~~config~~~~~')
        print(json.dumps(cfg, indent=2))
        print('~~~~~~~~~~~~~~~~')

        trainer.print_models()

    # train
    print_(rank, f'[INFO] training for {cfg.n_steps} steps...')
    while True:
        # fetch data
        if training_set_iterator is not None:
            with torch.autograd.profiler.record_function('data_fetch'):
                batch = next(training_set_iterator)
        else:
            batch = None

        # run batch through trainer
        trainer.run_batch(batch, batch_idx, cur_step)

        # update state
        cur_step += batch_size
        batch_idx += 1

        # check conditions
        done = (cur_step >= cfg.n_steps)
        do_eval = done or (cur_step % cfg.eval_freq) < batch_size
        do_maintenance = do_eval or \
            (cur_step % cfg.maintenance_freq) < batch_size

        if not do_maintenance:
            continue

        # print info
        tick_end_time = time.time()
        msg = 'step={}, steps/s={:.2f}'.format(
            cur_step,
            (cur_step - tick_start_step) / (tick_end_time - tick_start_time),
        )
        pad = 100 - len(msg)
        halfpad = pad // 2
        print_(rank, '~' * halfpad + msg + '~' * (pad - halfpad))
        print_(
            rank,
            '[TRAIN] maintenance={:.2f}, cpumem={:.2f}, gpumem={:.2f}'.format(
                maintenance_time,
                psutil.Process(os.getpid()).memory_info().rss / 2**30,
                torch.cuda.max_memory_allocated(device) / 2**30,
            ),
        )
        torch.cuda.reset_peak_memory_stats()

        if do_eval:
            # save
            print_(rank, '[INFO] saving model...')
            snapshot_data = {}
            for name, module in trainer.get_modules_for_save():
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(
                            module,
                            ignore_regex=r'.*\.w_avg',
                        )
                    module = copy.deepcopy(
                        module,
                    ).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            # resume_path = os.path.join(cfg.saves_dir, 'latest.pkl')
            resume_path = os.path.join(cfg.saves_dir, 'latest.pt')
            if rank == 0:
                # with open(resume_path, 'wb') as f:
                    # pickle.dump(snapshot_data, f)
                save_data = {
                    k: v.state_dict() for k, v in snapshot_data.items()
                }
                torch.save(save_data, resume_path)

            with torch.no_grad():
                # eval
                task.eval(trainer, snapshot_data, cur_step)

                # sample
                task.sample(trainer, snapshot_data, cur_step)

            del snapshot_data # conserve memory

        # collect stats
        trainer.collect_phase_stats()
        stats_collector.update()

        # print info
        if rank == 0:
            stats_dict = stats_collector.as_dict()
            for k, v in stats_dict.items():
                print('[TRAIN] {}: mean={:.4f}, std={:.4f}, n={}'.format(
                    k,
                    v['mean'],
                    v['std'],
                    v['num'],
                ))

        # update state
        tick_start_step = cur_step
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    print_(rank, 'exiting...')
