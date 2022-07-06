# *****************************************************************************
#  Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import functools
import json
import re
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.stats import betabinom

import common.layers as layers
from common.text.text_processing import TextProcessing
from common.utils import load_wav_to_torch, load_filepaths_and_text, to_gpu

from .acoustic_feat_extraction import estimate_pitch, estimate_energy 
from .cwt_conditioning import upsample_word_label


class BetaBinomialInterpolator:
    """Interpolates alignment prior matrices to save computation.

    Calculating beta-binomial priors is costly. Instead cache popular sizes
    and use img interpolation to get priors faster.
    """
    def __init__(self, round_mel_len_to=100, round_text_len_to=20):
        self.round_mel_len_to = round_mel_len_to
        self.round_text_len_to = round_text_len_to
        self.bank = functools.lru_cache(beta_binomial_prior_distribution)

    def round(self, val, to):
        return max(1, int(np.round((val + 1) / to))) * to

    def __call__(self, w, h):
        bw = self.round(w, to=self.round_mel_len_to)
        bh = self.round(h, to=self.round_text_len_to)
        ret = ndimage.zoom(self.bank(bw, bh).T, zoom=(w / bw, h / bh), order=1)
        assert ret.shape[0] == w, ret.shape
        assert ret.shape[1] == h, ret.shape
        return ret


def beta_binomial_prior_distribution(phoneme_count, mel_count, scaling=1.0):
    P = phoneme_count
    M = mel_count
    x = np.arange(0, P)
    mel_text_probs = []
    for i in range(1, M+1):
        a, b = scaling * i, scaling * (M + 1 - i)
        rv = betabinom(P, a, b)
        mel_i_prob = rv.pmf(x)
        mel_text_probs.append(mel_i_prob)
    return torch.tensor(np.array(mel_text_probs))


class TTSDataset(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self,
                 dataset_path,
                 audiopaths_and_text,
                 text_cleaners,
                 n_mel_channels,
                 symbol_set='english_basic',
                 p_arpabet=1.0,
                 n_speakers=1,
                 load_mel_from_disk=True,
                 load_pitch_from_disk=True,
                 load_cwt_from_disk=False,
                 cwt_accent=False,
                 pitch_mean=214.72203,  # LJSpeech defaults
                 pitch_std=65.72038,
                 max_wav_value=None,
                 sampling_rate=None,
                 filter_length=None,
                 hop_length=None,
                 win_length=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 prepend_space_to_text=False,
                 append_space_to_text=False,
                 pitch_online_dir=None,
                 betabinomial_online_dir=None,
                 use_betabinomial_interpolator=True,
                 pitch_online_method='pyin',
                 two_pass_method=False,
                 pitch_norm_method='default',
                 pitch_norm=True,
                 mels_downsampled=False, #If learning deidentified embeddings
                 load_ds_mel_from_disk=False, #If already extracted
                 **ignored):

        # Expect a list of filenames i.e. train and val
        if type(audiopaths_and_text) is str:
            audiopaths_and_text = [audiopaths_and_text]

        self.dataset_path = dataset_path
        self.audiopaths_and_text = load_filepaths_and_text(
            audiopaths_and_text, dataset_path,
            has_speakers=(n_speakers > 1)) #this now returns a list of dicts

        self.load_mel_from_disk = load_mel_from_disk
        if not load_mel_from_disk:
            self.max_wav_value = max_wav_value
            self.sampling_rate = sampling_rate
            self.stft = layers.TacotronSTFT(
                filter_length, hop_length, win_length,
                n_mel_channels, sampling_rate, mel_fmin, mel_fmax)

        self.mels_downsampled = mels_downsampled
        self.load_ds_mel_from_disk = load_ds_mel_from_disk
        if not load_ds_mel_from_disk and mels_downsampled:
            self.max_wav_value_ds = max_wav_value
            self.sampling_rate_ds = 800
            self.stft_ds = layers.TacotronSTFT(
                filter_length, 10, 40,
                n_mel_channels, self.sampling_rate_ds, mel_fmin, 400)

        self.load_pitch_from_disk = load_pitch_from_disk

        #word level conditioning
        self.cwt_accent = cwt_accent
        self.load_cwt_from_disk = load_cwt_from_disk

        self.prepend_space_to_text = prepend_space_to_text
        self.append_space_to_text = append_space_to_text

        assert p_arpabet == 0.0 or p_arpabet == 1.0, (
            'Only 0.0 and 1.0 p_arpabet is currently supported. '
            'Variable probability breaks caching of betabinomial matrices.')
        if self.cwt_accent == True:
            self.tp = TextProcessing(symbol_set, text_cleaners, p_arpabet=p_arpabet, get_counts=True) #get counts is getting number of symbols per word for upsampling
        else:
            self.tp = TextProcessing(symbol_set, text_cleaners, p_arpabet=p_arpabet, get_counts=False)
        
        self.n_speakers = n_speakers

        #Pitch extraction parameters
        self.pitch_tmp_dir = pitch_online_dir
        self.f0_method = pitch_online_method
        self.two_pass_method = two_pass_method
        self.norm_method = pitch_norm_method
        self.pitch_norm = pitch_norm
        self.betabinomial_tmp_dir = betabinomial_online_dir
        self.use_betabinomial_interpolator = use_betabinomial_interpolator

        if use_betabinomial_interpolator:
            self.betabinomial_interpolator = BetaBinomialInterpolator()

       # expected_columns = (2 + int(load_pitch_from_disk) + (n_speakers > 1))
       # @Johannah add a check for dictionary values to check for input arguments in meta
       # Add more asserts here.
        assert not (load_pitch_from_disk and self.pitch_tmp_dir is not None)


        to_tensor = lambda x: torch.Tensor([x]) if type(x) is float else x
        self.pitch_mean = to_tensor(pitch_mean)
        self.pitch_std = to_tensor(pitch_std)

    def __getitem__(self, index):

        #Indexing items using dictionary entries instead of list indices
        if self.n_speakers > 1:
            audiopath = self.audiopaths_and_text[index]['mels']
            text = self.audiopaths_and_text[index]['text']
            speaker = self.audiopaths_and_text[index]['speaker']
            speaker = int(speaker)
        else:
            audiopath = self.audiopaths_and_text[index]['mels']
            text = self.audiopaths_and_text[index]['text']
            speaker = None

        mel = self.get_mel(audiopath)
        text, text_info = self.get_text(text)
        pitch = self.get_pitch(index, mel.size(-1))
        energy = torch.norm(mel.float(), dim=0, p=2)
#        energy = estimate_energy(mel, norm=True, log=True)
       
        attn_prior = self.get_prior(index, mel.shape[1], text.shape[0])

        if self.cwt_accent == True:
            cwt_acc = self.get_cwt_labels(index, text_info)
        else:
            cwt_acc = None

        if self.mels_downsampled:
            audiopath = self.audiopaths_and_text[index]['mels_ds']
            ds_mel = self.get_ds_mel(audiopath)
            #print("DS MELSIZE,", ds_mel.size(-1))
            #assert ds_mel.size(-1) == mel.size(-1)
        else:
            ds_mel = None

        #print(pitch.size(-1), mel.size(-1),audiopath)
        assert pitch.size(-1) == mel.size(-1)
        assert energy.size(-1) == energy.size(-1)
        #print("MELSIZE, ", mel.size(-1))


        # No higher formants?
        if len(pitch.size()) == 1:
            pitch = pitch[None, :]

        return (text, mel, len(text), pitch, energy, speaker, attn_prior,
                audiopath, cwt_acc, ds_mel)

    def __len__(self):
        return len(self.audiopaths_and_text)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm,
                                                 requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.load(filename)
            # assert melspec.size(0) == self.stft.n_mel_channels, (
            #     'Mel dimension mismatch: given {}, expected {}'.format(
            #         melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_ds_mel(self, filename):
        #print(filename)
        if not self.load_ds_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
           # print(sampling_rate)
            if sampling_rate != self.stft_ds.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft_ds.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm,
                                                 requires_grad=False)
            melspec = self.stft_ds.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.load(filename)

        return melspec

    def get_text(self, text):
        if self.cwt_accent == True:
            text, text_info = self.tp.encode_text(text)
        else:
            text = self.tp.encode_text(text)
            text_info = None

        space = [self.tp.encode_text("A A")[1]]

        if self.prepend_space_to_text:
            text = space + text

        if self.append_space_to_text:
            text = text + space

        return torch.LongTensor(text), text_info

    def get_prior(self, index, mel_len, text_len):

        if self.use_betabinomial_interpolator:
            return torch.from_numpy(self.betabinomial_interpolator(mel_len,
                                                                   text_len))

        if self.betabinomial_tmp_dir is not None:
            audiopath, *_ = self.audiopaths_and_text[index]
            fname = Path(audiopath).relative_to(self.dataset_path) if self.dataset_path else Path(audiopath)
            fname = fname.with_suffix('.pt')
            cached_fpath = Path(self.betabinomial_tmp_dir, fname)

            if cached_fpath.is_file():
                return torch.load(cached_fpath)

        attn_prior = beta_binomial_prior_distribution(text_len, mel_len)

        if self.betabinomial_tmp_dir is not None:
            cached_fpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(attn_prior, cached_fpath)

        return attn_prior

    def get_pitch(self, index, mel_len=None):
        audiopath = self.audiopaths_and_text[index]['mels']
        if self.n_speakers > 1:
            spk = int(self.audiopaths_and_text[index]['speaker'])
        else:
            spk = 0

        if self.load_pitch_from_disk:
            pitchpath = self.audiopaths_and_text[index]['pitch']
            pitch = torch.load(pitchpath)

           #Johannah: I removed this for now but will fix later           
           # if self.pitch_mean is not None:
           #     assert self.pitch_std is not None
           #     pitch = normalize_pitch(pitch, self.pitch_mean, self.pitch_std)
            return pitch

        if self.pitch_tmp_dir is not None:
            fname = Path(audiopath).relative_to(self.dataset_path) if self.dataset_path else Path(audiopath)
            fname_method = fname.with_suffix('.pt')
            cached_fpath = Path(self.pitch_tmp_dir, fname_method)
            if cached_fpath.is_file():
                return torch.load(cached_fpath)

        # No luck so far - calculate
        wav = audiopath
        if not wav.endswith('.wav'):
            wav = re.sub('/mels/', '/wavs/', wav)
            wav = re.sub('.pt$', '.wav', wav)
        pitch_mel = estimate_pitch(wav, mel_len, self.f0_method, self.two_pass_method,
                                   self.pitch_mean, self.pitch_std, self.norm_method, self.pitch_norm)

        if self.pitch_tmp_dir is not None and not cached_fpath.is_file():
            cached_fpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pitch_mel, cached_fpath)

        return pitch_mel

    def get_cwt_labels(self, index, text_info, text=False):
        '''Reads in array of cwt on a per word basis and
           usamples OR predicts cwt labels and upsamples'''

        if self.load_cwt_from_disk:
            cwt_path = self.audiopaths_and_text[index]['cwt_accent']
            cwt_labels = torch.load(cwt_path)
            cwt_upsampled = upsample_word_label(text_info, cwt_labels)

            return torch.LongTensor(cwt_upsampled)

        else:
             #predict labels
             #Here we will eventually add the other task
            return None


class TTSCollate:
    """Zero-pads model inputs and targets based on number of frames per step"""

#For reference indices
#0 text
#1 mel
#2 len(text)
#3 pitch
#4 energy
#5 speaker
#6 attn_prior
#7 audiopath
#8 cwt_acc 
#9 ds_mel

    def __call__(self, batch):
        """Collate training batch from normalized text and mel-spec"""
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        # Include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        n_formants = batch[0][3].shape[0]
        pitch_padded = torch.zeros(mel_padded.size(0), n_formants,
                                   mel_padded.size(2), dtype=batch[0][3].dtype)
        energy_padded = torch.zeros_like(pitch_padded[:, 0, :])

        for i in range(len(ids_sorted_decreasing)):
            pitch = batch[ids_sorted_decreasing[i]][3]
            #print(pitch.shape)
            energy = batch[ids_sorted_decreasing[i]][4]
            pitch_padded[i, :, :pitch.shape[1]] = pitch
            energy_padded[i, :energy.shape[0]] = energy

        if batch[0][5] is not None:
            speaker = torch.zeros_like(input_lengths)
            for i in range(len(ids_sorted_decreasing)):
                speaker[i] = batch[ids_sorted_decreasing[i]][5]
        else:
            speaker = None

        attn_prior_padded = torch.zeros(len(batch), max_target_len,
                                        max_input_len)
        attn_prior_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            prior = batch[ids_sorted_decreasing[i]][6]
            attn_prior_padded[i, :prior.size(0), :prior.size(1)] = prior

        # Count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)

        audiopaths = [batch[i][7] for i in ids_sorted_decreasing]

        #if word-level conditioning pad cwt labels to max input length
        if batch[0][8] is not None:
            cwt_padded = torch.LongTensor(len(batch), max_input_len)
            cwt_padded.zero_()
            for i in range(len(ids_sorted_decreasing)):
                cwt = batch[ids_sorted_decreasing[i]][8]
                cwt_padded[i, :cwt.size(0)] = cwt
        else:
            cwt_padded = None
        #print(cwt_padded.size(), text_padded.size())

        #for downsampled mels

        # Right zero-pad mel-spec #check Johannah
        if batch[0][9] is not None:
            num_ds_mels = batch[0][9].size(0)
            max_target_len_ds = max([x[9].size(1) for x in batch])

#        if batch[0][9] is not None:
        # Include mel padded and gate padded
            ds_mel_padded = torch.FloatTensor(len(batch), num_ds_mels, max_target_len_ds)
            ds_mel_padded.zero_()
            output_lengths_ds = torch.LongTensor(len(batch))
            for i in range(len(ids_sorted_decreasing)):
                ds_mel = batch[ids_sorted_decreasing[i]][9]
                ds_mel_padded[i, :, :ds_mel.size(1)] = ds_mel
                output_lengths_ds[i] = ds_mel.size(1)
        else:
            ds_mel_padded = None

        return (text_padded, input_lengths, mel_padded, output_lengths, len_x,
                pitch_padded, energy_padded, speaker, attn_prior_padded,
                audiopaths, cwt_padded, ds_mel_padded, output_lengths_ds) #


def batch_to_gpu(batch):
    (text_padded, input_lengths, mel_padded, output_lengths, len_x,
     pitch_padded, energy_padded, speaker, attn_prior, audiopaths, cwt_padded, ds_mel_padded, output_lengths_ds) = batch

    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    mel_padded = to_gpu(mel_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    pitch_padded = to_gpu(pitch_padded).float()
    energy_padded = to_gpu(energy_padded).float()
    attn_prior = to_gpu(attn_prior).float()
    if speaker is not None:
        speaker = to_gpu(speaker).long()

    if cwt_padded is not None:
        cwt_padded = to_gpu(cwt_padded).long()

    if ds_mel_padded is not None:
        ds_mel_padded = to_gpu(ds_mel_padded).float()

    # Alignments act as both inputs and targets - pass shallow copies
    x = [text_padded, input_lengths, mel_padded, output_lengths,
         pitch_padded, energy_padded, speaker, attn_prior, audiopaths, cwt_padded, ds_mel_padded, output_lengths_ds]
    y = [mel_padded, input_lengths, output_lengths]
    len_x = torch.sum(output_lengths)
    return (x, y, len_x)
