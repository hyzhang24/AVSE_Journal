import os
from os.path import join, isfile
import torch
import logging
import random
from glob import glob
from decord import cpu
from decord import VideoReader
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io import wavfile
from scipy import signal
# from torchaudio import load
import numpy as np
import cv2
import librosa
import torch.nn.functional as F

windows = signal.windows.hann

# def activelev(data):
#     max_amp = np.std(data) 
#     if max_amp==0:
#         return 0
#     return data/max_amp

def normalize(samples, desired_rms = 0.1, eps = 1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return samples

def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class Specs(Dataset):
    def __init__(self, data_dir, subset, dummy, shuffle_spec, num_frames,
            format='lrs3', normalize="noisy", spec_transform=None, max_audio_len = 40800,
            max_video_len = 64, video_frame = (88, 88), stft_kwargs=None, **ignored_kwargs):

        # Read file paths according to file naming format.
        self.subset = subset
        self.max_a_len = max_audio_len
        self.max_v_len = max_video_len
        self.video_frame_size = video_frame
        if format == "lrs3":
            # self.train_root = join(data_dir, "train")
            # self.valid_root = join(data_dir, "dev")
            # self.test_root = join(data_dir, "test")
            if subset == "train":
                self.data_path = join(data_dir, "train")
                self.noisy_list = glob(join(self.data_path, "scenes", "*mixed.wav"))
                self.sample_num = len(self.noisy_list)
                self.clean_list = [x.replace("mixed", "target") for x in self.noisy_list]
                self.video_list = [x.replace("_mixed.wav", "_silent.mp4") for x in self.noisy_list]
                self.noise_list = [x.replace("mixed", "interferer") for x in self.noisy_list]
            elif subset == "dev":
                self.data_path = join(data_dir, "dev")
                self.noisy_list = glob(join(self.data_path, "scenes", "*mixed.wav"))
                self.clean_list = [x.replace("mixed", "target") for x in self.noisy_list]
                self.video_list = [x.replace("_mixed.wav", "_silent.mp4") for x in self.noisy_list]
                self.noise_list = [x.replace("mixed", "interferer") for x in self.noisy_list]
            elif subset == "test":
                self.data_path = join(data_dir, "test")
                self.noisy_list = glob(join(self.data_path, "scenes", "*mixed.wav"))
                self.clean_list = [x.replace("mixed", "target") for x in self.noisy_list]
                self.video_list = [x.replace("_mixed.wav", "_silent.mp4") for x in self.noisy_list]
                self.noise_list = None
            else: 
                raise NotImplementedError(f"invalid subset")
        else:
            raise NotImplementedError(f"Directory format {format} unknown!")

        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"

    def __getitem__(self, i):
        noisy = self.load_wav(self.noisy_list[i])
        clean = self.load_wav(self.clean_list[i])
        video_ = VideoReader(self.video_list[i], ctx=cpu(0))

        if self.shuffle_spec:
            # process audio data
            if clean.shape[0] > self.max_a_len:
                clip_idx = random.randint(0, clean.shape[0] - self.max_a_len)
                video_idx = int((clip_idx / 16000) * 25)
                clean = clean[clip_idx:clip_idx + self.max_a_len]
                noisy = noisy[clip_idx:clip_idx + self.max_a_len]
            else:
                video_idx = -1
                clean = np.pad(clean, pad_width=[0, self.max_a_len - clean.shape[0]], mode="constant")
                noisy = np.pad(noisy, pad_width=[0, self.max_a_len - noisy.shape[0]], mode="constant")
            # process video data
            if len(video_) < self.max_v_len:
                frames = video_.get_batch(list(range(len(video_)))).asnumpy()
            else:
                max_idx = min(video_idx + self.max_v_len, len(video_))
                frames = video_.get_batch(list(range(video_idx, max_idx))).asnumpy()
            bg_frames = [cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in range(len(frames))]
            bg_frames = np.array([cv2.resize(bg_frames[i], self.video_frame_size) for i in range(len(bg_frames))]).astype(
                np.float32)
            bg_frames /= 255.0
            if len(bg_frames) < self.max_v_len:
                bg_frames = np.concatenate(
                    (bg_frames,
                     np.zeros((self.max_v_len - len(bg_frames), self.video_frame_size[0], self.video_frame_size[1])).astype(bg_frames.dtype)),
                    axis=0)
        else: # if not random select patch from audio/video
            frames = video_.get_batch(list(range(len(video_)))).asnumpy()
            bg_frames = np.array(
                [cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in range(len(frames))]).astype(np.float32)
            bg_frames = np.array([cv2.resize(bg_frames[i], self.video_frame_size) for i in range(len(bg_frames))]).astype(
                np.float32)
            bg_frames /= 255.0
        # bg_frames = torch.tensor(bg_frames).unsqueeze(0)
        bg_frames = torch.tensor(bg_frames)
        normfac = np.max(np.abs(noisy))
        clean = torch.FloatTensor(clean / normfac)
        noisy = torch.FloatTensor(noisy / normfac)
        return self.get_audio_features(clean), self.get_audio_features(noisy), bg_frames

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.clean_list)/200)
        else:
            return len(self.clean_list)
    
    def load_wav(self, wav_path):
        return wavfile.read(wav_path)[1].astype(np.float32) / (2 ** 15)

    # def get_stft(self, audio):
        # return librosa.stft(audio, win_length=400, n_fft=512, hop_length=160, window="hann",
                            # center=True)

    def get_audio_features(self, audio):
        feat = self.spec_transform(torch.stft(audio, **self.stft_kwargs))
        return feat.unsqueeze(0)
    
    # @staticmethod
    # def get_files_list(data_root, test_set=False):
    #     files_list = []
    #     for file in os.listdir(join(data_root, "scenes")):
    #         if file.endswith("mixed.wav"):
    #             files = (join(data_root, "scenes", file.replace("mixed", "target")),
    #                         join(data_root, "scenes", file.replace("mixed", "interferer")),
    #                         join(data_root, "scenes", file),
    #                         join(data_root, "lips", file.replace("_mixed.wav", "_silent.mp4")),
    #                         file.replace("_mixed.wav", "")
    #                         )
    #             if not test_set:
    #                 if all([isfile(f) for f in files[:-1]]):
    #                     files_list.append(files)
    #             else:
    #                 files_list.append(files)
    #     return files_list


class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset. Should contain `train`, `valid` and `test` subdirectories, each of which contain `clean` and `noisy` subdirectories.")
        parser.add_argument("--format", type=str, choices=("lrs3", "dns"), default="lrs3", help="Read file paths according to file naming format.")
        parser.add_argument("--batch_size", type=int, default=8, help="The batch size. 8 by default.")
        parser.add_argument("--n_fft", type=int, default=510, help="Number of FFT bins. 510 by default.")   # to assure 256 freq bins
        parser.add_argument("--hop_length", type=int, default=128, help="Window hop length. 128 by default.")
        parser.add_argument("--num_frames", type=int, default=256, help="Number of frames for the dataset. 256 by default.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="The window function to use for the STFT. 'hann' by default.")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use for DataLoaders. 4 by default.")
        parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
        parser.add_argument("--spec_factor", type=float, default=0.15, help="Factor to multiply complex STFT coefficients by. 0.15 by default.")
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5, help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). 0.5 by default.")
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy", help="Normalize the input waveforms by the clean signal, the noisy signal, or not at all.")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent", help="Spectogram transformation for input representation.")
        return parser

    def __init__(
        self, base_dir="", format='lrs3', batch_size=8,
        n_fft=510, hop_length=128, num_frames=256, window='hann',
        num_workers=4, dummy=False, spec_factor=0.15, spec_abs_exponent=0.5,
        gpu=True, normalize='noisy', transform_type="exponent", **kwargs
    ):
        super().__init__()
        self.base_dir = base_dir
        self.format = format
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.kwargs = kwargs

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames,
            spec_transform=self.spec_fwd, **self.kwargs
        )
        if stage == 'fit' or stage is None:
            self.train_set = Specs(data_dir=self.base_dir, subset='train',
                dummy=self.dummy, shuffle_spec=True, format=self.format,
                normalize=self.normalize, **specs_kwargs)
            self.valid_set = Specs(data_dir=self.base_dir, subset='dev',
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, **specs_kwargs)
        if stage == 'test' or stage is None:
            self.test_set = Specs(data_dir=self.base_dir, subset='test',
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, **specs_kwargs)

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs()**e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window
    


    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})


    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )
    

