from torch.utils.data.dataset import Dataset


class BaseDataset(Dataset):
    @property
    def image_resolution(self):
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def image_channels(self):
        return self.image_shape[0]

    @property
    def image_shape(self):
        raise NotImplementedError('must specify image shape!')

    @property
    def description(self):
        raise NotImplementedError('must specify a description')


def init_dataset(dataset, args={}):
    from datasets.ship import Ship

    DATASETS = {'ship': Ship}

    try:
        DATASETS[dataset]
    except KeyError:
        raise KeyError(f'Unknown dataset type: {dataset}')

    return DATASETS[dataset](**args)