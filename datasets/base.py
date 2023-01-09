from torch.utils.data import Dataset
class DatasetBase(Dataset):
    def __init__(self, transforms, target_transforms):
        self.transform = transforms
        self.target_transform = target_transforms
        self.ways = self._take_ways()
        self.data, self.labels = self._read_data()

    def _read_data(self) -> tuple:
        raise NotImplementedError()

    def _take_ways(self) -> dict:
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):

        input_, target = self.data[index], self.labels[index]

        if self.transform is not None:
            input_ = self.transform(input_)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return input_, target