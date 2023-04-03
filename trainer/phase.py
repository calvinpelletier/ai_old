#!/usr/bin/env python3
import torch
import external.sg2.misc as misc


class TrainingPhase:
    def __init__(self, name, module, opt, interval, device, rank):
        self.name = name
        self.module = module
        self.opt = opt
        self.interval = interval
        self.device = device

        self.start_event = None
        self.end_event = None
        if rank == 0:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)


    def init_gradient_accumulation(self):
        if self.start_event is not None:
            self.start_event.record(torch.cuda.current_stream(self.device))
        self.opt.zero_grad(set_to_none=True)
        self.module.prep_for_train_phase()


    def update_params(self):
        self.module.requires_grad_(False)
        with torch.autograd.profiler.record_function(self.name + '_opt'):
            for param in self.module.parameters():
                if param.grad is not None:
                    misc.nan_to_num(
                        param.grad,
                        nan=0,
                        posinf=1e5,
                        neginf=-1e5,
                        out=param.grad,
                    )
            self.opt.step()
        if self.end_event is not None:
            self.end_event.record(torch.cuda.current_stream(self.device))
