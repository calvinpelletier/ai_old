#!/usr/bin/env python3
from ai_old.task.genderswap.static import StaticMtfTask
from torchvision.utils import save_image


class SsOnlyBlendUltTask(StaticMtfTask):
    def sampling_step(self, data, out, step):
        for id, male_rec, female_rec, mtf, ftm in zip(
            data['item_id'],
            out['male_rec'],
            out['female_rec'],
            out['mtf'],
            out['ftm'],
        ):
            save_image(
                [male_rec, mtf, female_rec, ftm],
                self.samples_path.format(step, id),
                normalize=True,
                range=(-1, 1),
            )

    def _get_fake_key(self):
        return 'female_rec'

    def _get_real_key(self):
        return 'female'


class BlendUltTask(StaticMtfTask):
    def sampling_step(self, data, out, step):
        for id, real_rec, real_swap, fake_gen, male_rec, female_rec, mtf, ftm in zip(
            data['item_id'],
            out['real_rec'],
            out['real_swap'],
            out['fake_gen'],
            out['male_rec'],
            out['female_rec'],
            out['mtf'],
            out['ftm'],
        ):
            save_image(
                [real_rec, real_swap, fake_gen, male_rec, mtf, female_rec, ftm],
                self.samples_path.format(step, id),
                normalize=True,
                range=(-1, 1),
            )

    def _get_fake_key(self):
        # return 'ss2_gen'
        return 'real_rec'
        # return 'mtf'

    def _get_real_key(self):
        # return 'ss2'
        return 'real'
        # return 'female'


class UltTask(StaticMtfTask):
    def sampling_step(self, data, out, step):
        for id, real_rec, real_swap, fake_gen, male_rec, female_rec, mtf, ftm in zip(
            data['item_id'],
            out['real_rec'],
            out['real_swap'],
            out['fake_gen'],
            out['male_rec'],
            out['female_rec'],
            out['mtf'],
            out['ftm'],
        ):
            save_image(
                [real_rec, real_swap, fake_gen, male_rec, mtf, female_rec, ftm],
                self.samples_path.format(step, id),
                normalize=True,
                range=(-1, 1),
            )

    def _get_fake_key(self):
        # return 'ss2_gen'
        return 'real_rec'
        # return 'mtf'

    def _get_real_key(self):
        # return 'ss2'
        return 'real'
        # return 'female'
