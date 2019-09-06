from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders.
    """
    def __init__(self, dataset, batch_size, shuffle, validation_ratio,
                 num_workers, collate_fn=default_collate):
        """
        Splits the dataset into training and validation by the given ratio
        and initialize this class with the training dataset splitted.

        If the ratio is 0.0, the dataset will not be splitted and this class
        will be initialized by the whole dataset given.
        """
        msg_ratio_invalid = 'Validation ratio should be between 0.0 and 1.0'
        assert 0.0 <= validation_ratio <= 1.0, msg_ratio_invalid

        n_samples = len(dataset)
        valid_set_len = round(validation_ratio * n_samples)
        train_set_len = n_samples - valid_set_len

        if valid_set_len > 0:
            self._train_dataset, self._valid_dataset = random_split(
                dataset, [train_set_len, valid_set_len])
        else:  # valid_set_len == 0
            self._train_dataset = dataset
            self._valid_dataset = None

        self._init_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }

        super().__init__(self._train_dataset, **self._init_kwargs)

    def get_validation_data_loader(self):
        """
        Returns a data loader of the validation dataset splitted.
        """
        if self._valid_dataset is None:
            return None
        else:
            return DataLoader(self._valid_dataset, **self._init_kwargs)
