#!/usr/bin/env python3
import os
import ai_old.constants as c
import torch.utils.data as data
from ai_old.dataset import metadata_manager
from ai_old.dataset.metadata_column_processor import MetadataColumnProcessor
import numpy as np
from ai_old.util.etc import print_

SET_LABEL_COL_NAME = "set_labels"


class DatasetBase(object):
    def filter_func(self):
        raise NotImplementedError("Implement filter_func in any subclasses of DatasetBase.")

    def select_cols(self):
        raise NotImplementedError("Implement select_cols in any subclasses of DatasetBase.")

    def test_set_label(self):
        return None

    def val_set_label(self):
        raise NotImplementedError("Implement val_set_label in any subclasses of DatasetBase.")

    def check_datasets(self, train, val, test):
        pass

    def get_column_processor_overrides(self):
        """If this is overridden, it should return a map of (column name -> column processor).

        See DEFAULT_COLUMN_PROCESSOR_MAP in metadata_column_processor.py for what this would look like.
        """
        return None

    def __init__(self, xflip, for_training=True):
        # TODO: another use case - different column for label in training vs. validation (e.g. generated gender float vs hand-labeled in validation)

        self.xflip = xflip

        columns_to_fetch = self.select_cols()
        if type(columns_to_fetch) is dict:
            columns_to_fetch = list(columns_to_fetch.keys())

        self.for_training = for_training

    def get_train_set(self, batch_size, seed, rank, num_gpus, verbose=True):
        metadata = metadata_manager.get().read_all_data(self.filter_func(),
                                                        list(self.select_cols().keys()) + ["item_id", SET_LABEL_COL_NAME])
        train_dataset = self.__create_training_set(metadata)
        if verbose:
            print_(rank, f'[DATASET] train size: {len(train_dataset)}')
        train_loader = self.__wrap_dataset_in_inf_loader(train_dataset, batch_size, seed, rank, num_gpus)
        return train_loader

    def get_val_set(self, batch_size, seed, rank, num_gpus, verbose=True):
        metadata = metadata_manager.get().read_all_data(self.filter_func(),
                                                        list(self.select_cols().keys()) + ["item_id", SET_LABEL_COL_NAME])
        val_dataset = self.__create_val_set(metadata)
        if verbose:
            print_(rank, f'[DATASET] val size: {len(val_dataset)}')
        val_loader = self.__wrap_dataset_in_loader(val_dataset, batch_size, seed, rank, num_gpus)
        return val_loader

    def get_test_set(self, batch_size, seed, rank, num_gpus, verbose=True):
        metadata = metadata_manager.get().read_all_data(self.filter_func(),
                                                        list(self.select_cols().keys()) + ["item_id", SET_LABEL_COL_NAME])
        test_dataset = self.__create_test_set(metadata)
        if verbose:
            print_(rank, f'[DATASET] test size: {len(test_dataset)}')
        test_loader = self.__wrap_dataset_in_loader(test_dataset, batch_size, seed, rank, num_gpus)
        return test_loader

    def get_clean_set(self, batch_size, seed, rank, num_gpus, verbose=True):
        metadata = metadata_manager.get().read_all_data(self.filter_func(),
                                                        list(self.select_cols().keys()) + ["item_id", SET_LABEL_COL_NAME])
        clean_dataset = self.__create_clean_set(metadata)
        if verbose:
            print_(rank, f'[DATASET] clean size: {len(clean_dataset)}')
        clean_loader = self.__wrap_dataset_in_loader(clean_dataset, batch_size, seed, rank, num_gpus)
        return clean_loader

    # def training_val_and_test(self, batch_size, seed, rank, num_gpus, verbose=True):
    #     metadata = metadata_manager.get().read_all_data(self.filter_func(),
    #                                                     list(self.select_cols().keys()) + ["item_id", SET_LABEL_COL_NAME])
    #
    #     train_dataset, val_dataset, test_dataset = self.__create_training_val_and_test(metadata)
    #     if verbose:
    #         print_(rank, f'[DATASET] train size: {len(train_dataset)}')
    #         print_(rank, f'[DATASET] val size: {len(val_dataset)}')
    #         print_(rank, f'[DATASET] test size: {len(test_dataset)}')
    #
    #     # Wrap the datasets in their respective loaders.
    #     args = [batch_size, seed, rank, num_gpus]
    #     train_loader = self.__wrap_dataset_in_loader(train_dataset, *args)
    #     val_loader = self.__wrap_dataset_in_loader(val_dataset, *args)
    #     test_loader = self.__wrap_dataset_in_loader(test_dataset, *args)
    #
    #     return train_loader, val_loader, test_loader

    def inference_no_data_loader(self):
        metadata = metadata_manager.get().read_all_data(self.filter_func(),
                                                        list(self.select_cols().keys()) + ["item_id"])

        full_dataset = self.__create_lasi_dataset(list(metadata.values()), is_training=False)
        return full_dataset

    def inference(self, batch_size, seed, rank, num_gpus):
        metadata = metadata_manager.get().read_all_data(self.filter_func(),
                                                        list(self.select_cols().keys()) + ["item_id"])

        full_dataset = self.__create_lasi_dataset(list(metadata.values()), is_training=False)
        return self.__wrap_dataset_in_loader(
            full_dataset,
            batch_size,
            seed,
            rank,
            num_gpus,
        )

    def has_training_set(self):
        return True

    def __wrap_dataset_in_inf_loader(self, ds, batch_size, seed, rank, num_gpus):
        sampler = DatasetBase.InfiniteSampler(
            dataset=ds,
            rank=rank,
            num_replicas=num_gpus,
            seed=seed,
        )
        return iter(data.DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=batch_size // num_gpus,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=2,
        ))

    def __wrap_dataset_in_loader(self, ds, batch_size, _seed, _rank, _num_gpus):
        return data.DataLoader(ds,
                               batch_size=batch_size,
                               shuffle=False,
                               drop_last=True,
                               num_workers=4)

    def __create_training_set(self, metadata):
        metadata_rows = list(metadata.values())
        train_md_rows = [obj for obj in metadata_rows if
                         self.test_set_label() not in obj.get(SET_LABEL_COL_NAME, []) and self.val_set_label() not in obj.get(SET_LABEL_COL_NAME, [])]
        train_dataset = self.__create_lasi_dataset(train_md_rows, is_training=True)
        return train_dataset

    def __create_val_set(self, metadata):
        metadata_rows = list(metadata.values())
        val_md_rows = [obj for obj in metadata_rows if
                        self.val_set_label() in obj.get(SET_LABEL_COL_NAME, [])] if self.val_set_label() else []
        val_dataset = self.__create_lasi_dataset(val_md_rows, is_training=False)
        return val_dataset

    def __create_test_set(self, metadata):
        metadata_rows = list(metadata.values())
        test_md_rows = [obj for obj in metadata_rows if
                        self.test_set_label() in obj.get(SET_LABEL_COL_NAME, [])] if self.test_set_label() else []
        test_dataset = self.__create_lasi_dataset(test_md_rows, is_training=False)
        return test_dataset

    def __create_clean_set(self, metadata):
        metadata_rows = list(metadata.values())
        clean_md_rows = [obj for obj in metadata_rows if
                        'clean' in obj.get(SET_LABEL_COL_NAME, [])]
        clean_dataset = self.__create_lasi_dataset(clean_md_rows, is_training=False)
        return clean_dataset

    # def __create_training_val_and_test(self, metadata):
    #     metadata_rows = list(metadata.values())
    #
    #     # Pull test data out of metadata.
    #     train_md_rows = [obj for obj in metadata_rows if
    #                      self.test_set_label() not in obj.get(SET_LABEL_COL_NAME, []) and self.val_set_label() not in obj.get(SET_LABEL_COL_NAME, [])]
    #     val_md_rows = [obj for obj in metadata_rows if self.val_set_label() in obj.get(SET_LABEL_COL_NAME, [])]
    #     test_md_rows = [obj for obj in metadata_rows if
    #                     self.test_set_label() in obj.get(SET_LABEL_COL_NAME, [])] if self.test_set_label() else []
    #
    #     # Create separate datasets for training and validation.
    #     train_dataset = self.__create_lasi_dataset(train_md_rows, is_training=True)
    #     val_dataset = self.__create_lasi_dataset(val_md_rows, is_training=False)
    #     test_dataset = self.__create_lasi_dataset(test_md_rows, is_training=False)
    #
    #     self.check_datasets(train_dataset, val_dataset, test_dataset)
    #
    #     return train_dataset, val_dataset, test_dataset

    # Just a helper method.
    def __create_lasi_dataset(self, metadata_rows, is_training):
        can_flip = is_training
        if not self.xflip:
            can_flip = False

        return DatasetBase.LasiDataset(metadata_rows, MetadataColumnProcessor(can_flip=can_flip,
                                                                              override_column_procs=self.get_column_processor_overrides()),
                                       column_name_mapping=self.select_cols())

    class LasiDataset(data.Dataset):
        def __init__(self, metadata_rows, metadata_column_processor, column_name_mapping):
            self.metadata_rows: list = metadata_rows
            self.mcp = metadata_column_processor
            self.column_name_mapping = column_name_mapping

            if len(self) == 0:
                raise RuntimeError(
                    "Length is 0 for created LasiDataset - maybe double check the filter functions you used when reading metadata?")

        def __getitem__(self, idx):
            row_dict = self.metadata_rows[idx]
            processed_row_dict = self.mcp.process_row(row_dict)
            return {
                self.column_name_mapping.get(k, k): v for k, v in processed_row_dict.items()
            }

        def __len__(self):
            return len(self.metadata_rows)

    class InfiniteSampler(data.Sampler):
        def __init__(self,
            dataset,
            rank=0,
            num_replicas=1,
            shuffle=True,
            seed=0,
            window_size=0.5,
        ):
            assert len(dataset) > 0
            assert num_replicas > 0
            assert 0 <= rank < num_replicas
            assert 0 <= window_size <= 1
            super().__init__(dataset)
            self.dataset = dataset
            self.rank = rank
            self.num_replicas = num_replicas
            self.shuffle = shuffle
            self.seed = seed
            self.window_size = window_size

        def __iter__(self):
            order = np.arange(len(self.dataset))
            rnd = None
            window = 0
            if self.shuffle:
                rnd = np.random.RandomState(self.seed)
                rnd.shuffle(order)
                window = int(np.rint(order.size * self.window_size))

            idx = 0
            while True:
                i = idx % order.size
                if idx % self.num_replicas == self.rank:
                    yield order[i]
                if window >= 2:
                    j = (i - rnd.randint(window)) % order.size
                    order[i], order[j] = order[j], order[i]
                idx += 1
