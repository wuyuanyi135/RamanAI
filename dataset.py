import torch
import torch.utils
import torch.utils.data


class RamanDataset(torch.utils.data.Dataset):
    def __init__(self, input, target=None, normalize=True, generate_original=True):
        super().__init__()
        self.generate_original = generate_original
        self.input_mean = None
        self.input_std = None

        self.target_mean = None
        self.target_std = None

        self.input = torch.tensor(input, dtype=torch.float32)
        if target is not None:
            self.target = torch.tensor(target, dtype=torch.float32)
        else:
            self.target = None

        if normalize:
            self.input, self.input_mean, self.input_std = RamanDataset.normalize_(self.input)
            self.target, self.target_mean, self.target_std = RamanDataset.normalize_(self.target)

    def __getitem__(self, idx):
        if self.target is not None:
            input, target = self.input[idx, :], self.target[idx, :]
            if self.generate_original:
                input = input * self.input_std + self.input_mean
                target = target * self.target_std + self.target_mean
            return input, target
        else:
            input = self.input[idx, :]
            if self.generate_original:
                input = input * self.input_std + self.input_mean
            return input

    @staticmethod
    def normalize_(data: torch.Tensor):
        mean = data.mean(0)
        std = data.std(0)

        return (data - mean) / std, mean, std

    def normalize_with(self, input_mean, input_std, target_mean, target_std):
        self.input = (self.input - input_mean)/input_std
        if self.target is not None:
            self.target = (self.target - target_mean)/target_std
        self.target_std = target_std
        self.target_mean = target_mean
        self.input_mean = input_mean
        self.input_std = input_std
    def __len__(self):
        return self.input.shape[0]
