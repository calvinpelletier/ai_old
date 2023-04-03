import os
import pickle
from collections import namedtuple

import ai_old.constants as c

DatasetItemId = namedtuple("DatasetItemId", "dataset item_id")

# Map of <column name> -> <file extension on disk>
SUPPLEMENTAL_DATA_EXTENSIONS = {
    "face_image_256": "png",
    "face_image_512": "png",
    "face_image_1024": "png",
    "segmented_face": "npy",
    "z": "npy",
    "fg": "png",
    "dilated_fg_mask": "npy",
    "ibg": "png",
    "soft_bg_mask": "npy",
    "soft_bg_mask_256": "npy",
    "clip_attrs": "npy",
    "e4e_inv_w_plus": "npy",
    "e4e_inv_swap_w_plus": "npy",
    "e4e_inv_clip": "npy",
    "e4e_inv_1024": "png",
    "e4e_inv_256": "png",
    "e4e_inv_128": "png",
    "e4e_inv_swap_256": "png",
    "e4e_inv_swap_128": "png",
    "outer_1536": "png",
    "outer_512": "png",
    "outer_384": "png",
    "outer_192": "png",
    "fhbc_128": "npy",
    "e4e_inv_seg_128": "npy",
    "outer_fhbc_seg_384": "npy",
    "outer_fhbc_seg_192": "npy",
    "inpaint_mask_512": "png",
    'fg_mask_256': 'png',
    # "enc_4x4_base": "pt",
    # "enc_4x4_guide": "pt",
    "enc_4x4_target": "pt",

    # synthswap
    "static_ss": "png",
    "dynamic_ss": "png",
    "dynamic_ss_seg": "npy",
    "dynamic_ss_fg": "png",
    "ss_dilated_fg_mask": "npy",
    "ss_ibg": "png",
}

DEFAULT_COLUMN_VALUES = {
    "test_set": [],
    "set_labels": set()
}


# TODO: considerations:
# - consider using some kind of schema to reduce size compared to storing dicts for every single row.
# - TODO: set_working_dataset function?


def get():
    """
    Helper method to read metadata from its default location. Note that this is potentially slow as it's a big IO read.
    It shouldn't be called within any large loops.
    """
    return MetadataManager.from_file(create_if_nonexistent=False)


class MetadataManager(object):
    def __init__(self):
        self.data = {}

    @staticmethod
    def from_file(override_path=None, create_if_nonexistent=False):
        inpath = override_path
        if inpath is None:
            inpath = os.path.join(c.ASI_DATASETS_PATH, "metadata")

        result = MetadataManager()
        if os.path.exists(inpath):
            result.data = pickle.load(open(inpath, "rb"))
        elif create_if_nonexistent:
            result.save(override_path=inpath)
        else:
            raise RuntimeError("Metadata file at %s doesn't exist!" % inpath)

        return result

    def read_all_data(self, row_filter_func=None, select_cols=None):
        """
        :param row_filter_func: Optional, f(dict) -> bool. Rows are included where function returns True.
        :param select_cols: Optional, list of string column names.
        :return: Dictionary of {column_id -> column value dict}
        """

        selected_rows = self.data
        if row_filter_func:
            selected_rows = {k: v for k, v, in selected_rows.items()
                             if row_filter_func(v)}

        final_result: dict = selected_rows
        if select_cols:
            if type(select_cols) is str:
                select_cols = [select_cols]
            select_cols = set(select_cols)
            final_result = {k: self.__filter_keys_in_dict(v, select_cols, require=True)
                            for k, v in final_result.items()}

        return final_result

    def __filter_keys_in_dict(self, val_dict, select_keys: set, require=True):
        result = {
            k: v for k, v in val_dict.items() if k in select_keys
        }

        # Substitute default values if any keys are missing.
        for k in select_keys:
            if k not in result and k in DEFAULT_COLUMN_VALUES:
                result[k] = DEFAULT_COLUMN_VALUES[k]

        if require and set(result.keys()) != select_keys:
            print(val_dict)
            raise RuntimeError("Requested columns %s but got %s" % (str(select_keys), str(result.keys())))

        return result

    def read_single_value(self, dataset, item_id, col_name):
        """Error if row doesn't exist, returns None if col doesn't exist for row."""
        k = self.__key(dataset, item_id)
        if not k in self.data:
            raise RuntimeError("Row %s does not exist" % str(k))
        val_dict = self.get(dataset, item_id)

        default_val = DEFAULT_COLUMN_VALUES.get(col_name, None)
        return val_dict.get(col_name, default_val)

    def get(self, dataset, item_id):
        k = self.__key(dataset, item_id)
        if not k in self.data:
            raise RuntimeError("Row %s does not exist" % str(k))

        return self.data[k]

    def contains(self, dataset, item_id):
        k = self.__key(dataset, item_id)
        return k in self.data

    def add_row(self, dataset, item_id, **other_columns):
        k = self.__key(dataset, item_id)
        if k in self.data:
            raise RuntimeError("Row with id %s already exists" % str(k))
        self.data[k] = {'dataset': dataset,
                        'item_id': item_id,
                        **other_columns}

    def add_or_update_columns(self, dataset, item_id, **columns_to_update):
        k = self.__key(dataset, item_id)
        if not k in self.data:
            raise RuntimeError("Row %s does not exist" % str(k))

        for key, val in columns_to_update.items():
            self.data[k][key] = val

    def clear_items(self, dataset):
        self.data = {k: v for k, v in self.data.items() if k.dataset != dataset}

    def clear_column(self, col_name):
        def rm_col(val_dict, c):
            return {k: v for k, v in val_dict.items() if k != c}

        self.data = {k: rm_col(v, col_name) for k, v in self.data.items()}

    def get_write_path_for_data(self, dataset, item_id, col_name):
        # TODO: fancier things in this method

        if col_name not in SUPPLEMENTAL_DATA_EXTENSIONS:
            raise RuntimeError("Unknown column name for supplemental data: ", col_name)

        rel_path = os.path.join(
            c.SUPPLEMENTAL_DATASET_FOLDER_NAME,
            col_name,
            dataset,
            item_id + "." + SUPPLEMENTAL_DATA_EXTENSIONS[col_name],
        )

        # Function to be used by caller when file write to disk is complete. Adds the files (relative) path as a column
        # in metadata.
        def mark_done_func():
            self.add_or_update_columns(dataset, item_id, **{col_name: rel_path})

        abs_path = os.path.join(c.ASI_DATASETS_PATH, rel_path)

        dir_path = os.path.dirname(abs_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        return MetadataManager.PendingSupplementalWrite(abs_path, mark_done_func)

    def num_items(self):
        return len(self.data)

    def __key(self, dataset, item_id):
        return DatasetItemId(dataset, item_id)  # "%s_%s" % (dataset, item_id)

    def save(self, override_path=None):
        """

        :param override_path: Path to write to (defaults to 'metadata' inside envvar ASI_DATASETS)
        :return:
        """
        outpath = override_path
        if outpath is None:
            outpath = os.path.join(c.ASI_DATASETS_PATH, "metadata")
        if os.path.exists(outpath):
            print("Warning: overwriting existing metadata file!")
        pickle.dump(self.data, open(outpath, "wb"))

    # Just for debugging.
    def print_all_column_names(self):
        column_names = set([col_name for val_dict in self.data.values() for col_name in val_dict.keys()])
        print(", ".join(column_names))

    class PendingSupplementalWrite(object):
        def __init__(self, abs_path, mark_done_func):
            self.abs_path = abs_path
            self.mark_done = mark_done_func


def id_to_fname(dataset, item_id):
    """
    :param dataset_item_id: Encapsulates the dataset name as well as the item ID from the metadata file.
    """
    if dataset == "ffhq-128":
        return os.path.join(c.ASI_DATASETS_PATH, dataset, "x", item_id + ".png")
    else:
        raise RuntimeError("Unsupported dataset: %s" % dataset)


def fname_to_id(dataset, rel_fname):
    """
    :param dataset: Dataset name, e.g. "ffhq-128". Corresponds to a folder within datasets base folder.
    :param rel_fname: Path to the file relative to the specific dataset folder root.
    :return: An item ID (string)
    """

    # TODO: maybe do this per-dataset.
    return os.path.splitext(os.path.basename(rel_fname))[0]
