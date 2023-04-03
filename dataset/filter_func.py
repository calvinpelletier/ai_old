"""
Helper functions for filtering metadata reads.
"""


def for_dataset(dataset_name, additional_filter=None):
    is_member_of_dataset = lambda obj: obj['dataset'] == dataset_name

    if additional_filter:
        return lambda obj: is_member_of_dataset(obj) and additional_filter(obj)

    return is_member_of_dataset

# class FilterFunc():
#
#     def __init__(self, func):
#         self.func = func
#
#     @staticmethod
#     def for_dataset(dataset_name, additional_filter=None):
#         is_member_of_dataset = lambda obj: obj['dataset'] == dataset_name
#
#         if additional_filter:
#             return lambda obj: is_member_of_dataset(obj) and additional_filter(obj)
#
#         return is_member_of_dataset
#
#     def with_additional(self, additional_filter):
#         new_func = lambda a
