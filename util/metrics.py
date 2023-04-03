#!/usr/bin/env python3
from ai_old.util.factory import build_dataset
import copy
from ai_old.metrics.metric_utils import FeatureStats, get_feature_detector
import torch


def calc_manifold_stats_for_generator(
    opts,
    detector_url,
    detector_kwargs,
    rel_lo=0,
    rel_hi=1,
    batch_size=64,
    batch_gen=None,
    jit=False,
    **stats_kwargs,
):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # build a new training dataset
    dataset_core = build_dataset(opts.cfg.dataset)
    training_set_iterator = dataset_core.get_train_set(
        batch_gen,
        0, # seed
        opts.rank,
        opts.num_gpus,
        verbose=False,
    )

    # setup model
    model = copy.deepcopy(
        opts.model,
    ).eval().requires_grad_(False).to(opts.device)

    assert not jit

    # init
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(
        tag='generator features',
        num_items=stats.max_items,
        rel_lo=rel_lo,
        rel_hi=rel_hi,
    )
    detector = get_feature_detector(
        url=detector_url,
        device=opts.device,
        num_gpus=opts.num_gpus,
        rank=opts.rank,
        verbose=progress.verbose,
    )

    # main loop
    while not stats.is_full():
        images = []
        for _i in range(batch_size // batch_gen):
            batch = next(training_set_iterator)
            images.append(opts.eval_fn(model, batch, batch_gen, opts.device))
        features = detector(torch.cat(images), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    return stats
