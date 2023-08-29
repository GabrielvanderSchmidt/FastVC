import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision
from torch.utils.data import Dataset, DataLoader


class FlowFromDirectory:
    def __init__(self, root_dir: str, get_all: bool = False, formats: tuple = (".jpg",)):
        r"""
        Searches, lists, and labels files in subdirectories of a specified root dir.

        Args:
            root_dir (str): Path to root directory of the dataset
            get_all (bool, optional): Gets files from the entire directory tree if True, only from root_dir sub-dirs if False.
            formats (tuple[str], optional): Tuple of file formats to include.
        """
        self.root_dir = root_dir
        self.get_all = get_all
        self.formats = formats

    def __call__(self) -> tuple:
        r"""
        Returns:
             X (list): Paths to files found.
             y (list): File labels. Sub-dirs of root_dir from where each file was taken from
        """
        X = []
        y = []
        for subdir in os.listdir(self.root_dir):
            subdir_path = os.path.join(self.root_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            if self.get_all is False:
                files = os.listdir(subdir_path)
            else:
                _, _, files = os.walk(subdir_path)
            for file in files:
                file_path = os.path.join(subdir_path, file)
                if file.endswith(self.formats) and os.path.isfile(file_path):
                    X.append(file_path)
                    y.append(subdir)
        return X, y


class ImageReadWrapper:
    def __init__(self, mode: torchvision.io.ImageReadMode = torchvision.io.ImageReadMode.UNCHANGED):
        r"""
        Wrapper for TorchVision's ``io.read_image`` function.

        Args:
             mode (ImageReadMode, optional): Read mode for `read_image``. Default ``ImageReadMode.UNCHANGED``.
        """
        self.mode = mode

    def __call__(self, path):
        return torchvision.io.read_image(path, self.mode)


class SiamesePairDataset(Dataset):
    def __init__(self, file_paths: list, labels: list, file_loader, augmentation=None, same_class_proba=None,
                 load_on_demand: bool = True):
        assert len(file_paths) == len(labels)
        super().__init__()
        self.data = file_paths
        self.labels = labels
        self.file_loader = file_loader
        self.augmentation = augmentation
        self.on_demand = load_on_demand

        # Divide samples according to their class (to help generate same/different class pairs)
        self.class_indexes = dict()
        for i in range(len(self.labels)):
            if self.labels[i] not in self.class_indexes.keys():
                self.class_indexes[self.labels[i]] = [i]
            else:
                self.class_indexes[self.labels[i]].append(i)
        # So we don't have to calculate it over and over again in __getitem__
        self.class_indexes_len = dict([(label, len(self.class_indexes[label])) for label in self.class_indexes.keys()])

        # Infer same-class pair probability from class frequency (as if directly sampling from ``self.data``)
        self.same_class_proba: dict
        if same_class_proba is None:
            self.same_class_proba = dict([(label, self.class_indexes_len[label] / len(self.data)) for label in self.class_indexes.keys()])

        elif isinstance(same_class_proba, float):  # Equal same-class pair probability for all classes
            if not (0 < same_class_proba < 1):
                raise ValueError(f"Same-class pair probability must be in the interval (0, 1), received {same_class_proba}.")
            self.same_class_proba = dict([(label, same_class_proba) for label in self.class_indexes.keys()])

        elif isinstance(same_class_proba, dict):  # Same-class probability of each class specified by dictionary
            if same_class_proba.keys() != self.class_indexes.keys():
                raise ValueError("Classes from 'labels' and 'same_class_proba' don't match.")
            if not all(0 < proba < 1 for proba in same_class_proba.values()):
                err_ = [i for i in same_class_proba.values() if not (0 < i < 1)][0]
                raise ValueError(f"Same-class pair probability must be in the interval (0, 1), received {err_}.")
            self.same_class_proba = same_class_proba

        else:
            raise TypeError(f"'same_class_proba' expects a float, a dict, or None, got type: {type(same_class_proba)}.")

        if not self.on_demand:  # Keep all data in memory
            for i, file_path in enumerate(self.data):
                self.data[i] = file_loader(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idxA = idx
        idxB = None
        ya = self.labels[idxA]
        if random.random() <= self.same_class_proba[ya]:
            idxB = random.choice(self.class_indexes[ya])
        else:
            allowed_classes = list(self.class_indexes.keys())
            allowed_classes.remove(ya)
            # Calculate class probabilities as if directly sampling from ``self.data`` (excluding class ``ya``)
            data_len = len(self.data) - self.class_indexes_len[ya]  # Make probabilities add up to 1
            probabilities = [self.class_indexes_len[i] / data_len for i in allowed_classes]
            cum_probabilities = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]

            rand_value = random.random()
            for prob, label in zip(cum_probabilities, allowed_classes):
                if rand_value <= prob:
                    idxB = random.choice(self.class_indexes[label])
                    break

        yb = self.labels[idxB]

        Xa = self.file_loader(self.data[idxA]) if self.on_demand else self.data[idxA]
        Xb = self.file_loader(self.data[idxB]) if self.on_demand else self.data[idxB]
        if self.augmentation is not None:
            Xa = self.augmentation(Xa)
            Xb = self.augmentation(Xb)
        y = 1 if ya == yb else 0
        return Xa, Xb, y


class SpectrogramPairDataset(Dataset):
    def __int__(self, root_dir: str = "./", transform=None, same_class_proba: float = 0.5):
        """
            Custom Dataset class to generate spectrogram image pairs from directory with audio files.
        :param root_dir (string): Root directory of the dataset.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.same_class_proba = same_class_proba
        self.samples = []
        self.class_ranges = {}
        self._get_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Get pair indexes
        sampleA = self.samples[index]
        if random.random() <= self.same_class_proba:
            sampleB = self.samples[random.randint(*self.class_ranges[sampleA[1]])]
            while sampleB == sampleA:
                sampleB = self.samples[random.randint(*self.class_ranges[sampleA[1]])]
        else:
            sampleB = self.samples[random.randint(0, len(self))]
            while self.class_ranges[sampleA[1]][0] <= sampleB[0] <= self.class_ranges[sampleA[1]][1]:
                sampleB = self.samples[random.randint(0, len(self))]
        # Load audio samples
        audioA, _ = torchaudio.load(sampleA[0])
        audioB, _ = torchaudio.load(sampleB[0])

        # Apply transforms and return spectrogram
        if not self.transform is None:
            return self.transform(audioA), self.transform(audioB)
        else:
            return audioA, audioB

    def _get_samples(self):
        start = end = 0
        for speaker in os.listdir(self.root_dir):
            speaker_path = os.path.join(self.root_dir, speaker)
            if not os.path.isdir(speaker_path):
                continue
            for chapter in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter)
                if not os.path.isdir(chapter_path):
                    continue
                for file in os.listdir(chapter_path):
                    file_path = os.path.join(chapter_path, file)
                    if not file_path.endswith(".flac"):
                        continue
                    self.samples.append((file_path, int(speaker)))
                    end += 1
            self.class_ranges[int(speaker)] = (start, end - 1)
            start = end


# Done with the help of
# https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/siamese_network/main.py
class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # weights = ResNet18_Weights.DEFAULT
        self.resnet = torchvision.models.resnet18(weights="DEFAULT", progress=True)
        self.resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                      bias=False)
        self.out_features = self.resnet.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        return x


class SiameseNetwork(torch.nn.Module):
    def __init__(self, feat_extractor: FeatureExtractor):
        super().__init__()
        self.feat_extractor = feat_extractor
        self.fc_in_features = self.feat_extractor.out_features
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.fc_in_features * 2, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 1),
        )
        #self.sigmoid = nn.Sigmoid()

    def forward(self, imageA: torch.Tensor, imageB: torch.Tensor) -> torch.Tensor:
        deep_featuresA = self.feat_extractor(imageA)
        deep_featuresB = self.feat_extractor(imageB)
        output = torch.cat((deep_featuresA, deep_featuresB), 1)

        output = self.fc(output)
        #output = self.sigmoid(output)
        return output


class Trainer:
    def __init__(self, model, loss_fn, optimizer, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.data = None

    def epoch(self, idx):
        total_loss = 0
        for batch_idx, (Xa, Xb, y) in enumerate(self.data):
            Xa, Xb, y = Xa.float().to(self.device), Xb.float().to(self.device), y.unsqueeze(-1).float().to(self.device)
            #Xa.to(self.device)
            #Xb.to(self.device)
            #y.to(self.device)
            y_hat = model.forward(Xa, Xb)
            loss = self.loss_fn(y_hat, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * Xa.size(0)
        print(f"Epoch {idx}\t loss = {(total_loss / len(self.data.dataset)):>7f}")

    def fit(self, dataloader, epochs):
        self.data = dataloader
        for i in range(epochs):
            self.model.train()
            self.epoch(i)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    flow = FlowFromDirectory(root_dir=r"./LibriSpeech/spectrograms", formats=(".png",))
    X, y = flow()
    print(len(X), len(y))

    # Temporary solution
    #train_split = int(0.8 * len(X))
    #tmp = list(zip(X, y))
    #random.seed(42)
    #random.shuffle(tmp)
    #X, y = zip(*tmp)
    #X_train, X_val = (X[:train_split]), list(X[train_split:])
    #y_train, y_val = list(y[:train_split]), list(y[train_split:])

    dataset = SiamesePairDataset(file_paths=X, labels=y,
                                 file_loader=ImageReadWrapper(mode=torchvision.io.ImageReadMode.GRAY),
                                 load_on_demand=False)
    batch_size = 256
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    extractor = FeatureExtractor()
    model = SiameseNetwork(feat_extractor=extractor)
    model.to(device)
    loss = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters())

    train = Trainer(model=model, loss_fn=loss, optimizer=optim, device=device)
    train.fit(dataloader=dataloader, epochs=10)

    print("Training ended, evaluating performance...")
    model.eval()
    total_loss = 0
    for batch_idx, (Xa, Xb, y) in enumerate(dataloader):
        Xa, Xb, y = Xa.float().to(train.device), Xb.float().to(train.device), y.unsqueeze(-1).float().to(train.device)
        y_hat = model.forward(Xa, Xb)
        loss = train.loss_fn(y_hat, y)
        total_loss += loss.item() * Xa.size(0)
        correct = 0
        for pred, answer in zip(torch.squeeze(y_hat).detach().numpy(), torch.squeeze(y).detach().numpy()):
            if (answer == 0 and pred < 0.5) or (answer == 1 and pred >= 0.5):
                correct += 1
    print(f"Epoch {idx}\t acc = {correct / len(torch.squeeze(y).detach().numpy()):>7f}\t loss = {(total_loss / len(dataloader.dataset)):>7f}")
