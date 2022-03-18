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

import re
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

import common.layers as layers
from common.text import cmudict
from common.text.text_processing import TextProcessing
from common.utils import load_wav_to_torch, load_filepaths_and_text, to_gpu
from tgt.io import read_textgrid


def check_durations(durs, mel_len, filepath):
    assert sum(durs) == mel_len, \
            f'Length mismatch: {filepath}, {sum(durs)} durs != {mel_len} lens'


def parse_textgrid(tier, sampling_rate, hop_length):
    # From Dan Wells
    # Latest MFA replaces silence phones with "" in output TextGrids
    sil_phones = ['sil', 'sp', 'spn', '']
    start_time = tier[0].start_time
    end_time = tier[-1].end_time
    phones = []
    durations = []
    for index, label in enumerate(tier._objects):
        p_start, p_end, phone = label.start_time, label.end_time, label.text
        # if p_start > end_time:
        #     phones.append('')
        end_time = p_end
        if phone not in sil_phones:
            phones.append(phone)
        else:
            if (index == 0) or (index == len(tier) - 1):
                # leading or trailing silence
                phones.append('sil')
            else:
                # short pause between words
                phones.append('sp')

        durations.append(int(np.ceil(p_end * sampling_rate / hop_length)
                             - np.ceil(p_start * sampling_rate / hop_length)))

    return phones, durations, start_time, end_time


def estimate_pitch(wav, mel_len, method='pyin', normalize_mean=None,
                   normalize_std=None, n_formants=1):

    if type(normalize_mean) is float or type(normalize_mean) is list:
        normalize_mean = torch.tensor(normalize_mean)

    if type(normalize_std) is float or type(normalize_std) is list:
        normalize_std = torch.tensor(normalize_std)

    if method == 'pyin':

        snd, sr = librosa.load(wav)
        pitch_mel, voiced_flag, voiced_probs = librosa.pyin(
            snd, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), frame_length=1024)
        assert np.abs(mel_len - pitch_mel.shape[0]) <= 1.0

        pitch_mel = np.where(np.isnan(pitch_mel), 0.0, pitch_mel)
        pitch_mel = torch.from_numpy(pitch_mel).unsqueeze(0)
        pitch_mel = F.pad(pitch_mel, (0, mel_len - pitch_mel.size(1)))

        if n_formants > 1:
            raise NotImplementedError

    else:
        raise ValueError

    pitch_mel = pitch_mel.float()

    if normalize_mean is not None:
        assert normalize_std is not None
        pitch_mel = normalize_pitch(pitch_mel, normalize_mean, normalize_std)

    return pitch_mel


def normalize_pitch(pitch, mean, std):
    zeros = (pitch == 0.0)
    pitch -= mean[:, None]
    pitch /= std[:, None]
    pitch[zeros] = 0.0
    return pitch


class TTSDataset(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, dataset_path, audiopaths_and_text, text_cleaners,
                 n_mel_channels, symbol_set='english_basic', p_arpabet=1.0,
                 cmu_dict='cmudict/cmudict-0.7b',
                 n_speakers=1, load_mel_from_disk=True,
                 load_pitch_from_disk=True, pitch_mean=214.72203,
                 pitch_std=65.72038, energy_mean=51.796032, energy_std=9.861213,
                 max_wav_value=None, sampling_rate=None,
                 filter_length=None, hop_length=None, win_length=None,
                 mel_fmin=None, mel_fmax=None, prepend_space_to_text=False,
                 append_space_to_text=False, load_durs_from_disk=False,
                 dur_online_dir=None, textgrid_path=None,
                 pitch_online_dir=None, pitch_online_method='pyin', **ignored):

        # Expect a list of filenames
        if type(audiopaths_and_text) is str:
            audiopaths_and_text = [audiopaths_and_text]

        self.hop_length = hop_length
        self.dataset_path = dataset_path
        self.textgrid_path = textgrid_path
        self.audiopaths_and_text = load_filepaths_and_text(
            audiopaths_and_text, dataset_path,
            has_speakers=(n_speakers > 1))
        self.load_mel_from_disk = load_mel_from_disk
        if not load_mel_from_disk:
            self.max_wav_value = max_wav_value
            self.sampling_rate = sampling_rate
            self.stft = layers.TacotronSTFT(
                filter_length, hop_length, win_length,
                n_mel_channels, sampling_rate, mel_fmin, mel_fmax)
        self.load_pitch_from_disk = load_pitch_from_disk
        self.load_durs_from_disk = load_durs_from_disk

        self.prepend_space_to_text = prepend_space_to_text
        self.append_space_to_text = append_space_to_text

        assert p_arpabet == 0.0 or p_arpabet == 1.0, (
            'Only 0.0 and 1.0 p_arpabet is currently supported. '
            'Variable probability breaks caching of betabinomial matrices.')
        if p_arpabet > 0.0:
            cmudict.initialize(cmu_dict, keep_ambiguous=True)

        self.tp = TextProcessing(symbol_set, text_cleaners, p_arpabet=p_arpabet, handle_arpabet='word', handle_arpabet_ambiguous='random')
        self.n_speakers = n_speakers
        self.pitch_tmp_dir = pitch_online_dir
        self.dur_tmp_dir = dur_online_dir
        self.f0_method = pitch_online_method

        expected_columns = (2 + int(load_durs_from_disk) + int(load_pitch_from_disk) + (n_speakers > 1))
        assert not (load_pitch_from_disk and self.pitch_tmp_dir is not None)

        if len(self.audiopaths_and_text[0]) < expected_columns:
            raise ValueError(f'Expected {expected_columns} columns in audiopaths file. '
                             'The format is <mel_or_wav>|[<pitch>|]<text>[|<speaker_id>]')

        if len(self.audiopaths_and_text[0]) > expected_columns:
            print('WARNING: Audiopaths file has more columns than expected')

        to_tensor = lambda x: torch.Tensor([x]) if type(x) is float else x
        self.pitch_mean = to_tensor(pitch_mean)
        self.pitch_std = to_tensor(pitch_std)
        self.energy_mean = to_tensor(energy_mean)
        self.energy_std = to_tensor(energy_std)

    def __getitem__(self, index):
        # Separate filename and text
        if self.n_speakers > 1:
            audiopath, *extra, text, speaker = self.audiopaths_and_text[index]
            speaker = int(speaker)
        else:
            audiopath, *extra, text = self.audiopaths_and_text[index]
            speaker = None

        mel = self.get_mel(audiopath)
        pitch = self.get_pitch(index, mel.size(-1))
        energy = torch.norm(mel.float(), dim=0, p=2)
        if self.energy_mean is not None:
            assert self.energy_std is not None
            norm_energy = normalize_pitch(energy.unsqueeze(dim=0), self.energy_mean, self.energy_std)
            energy = norm_energy.squeeze()

        dur, phones = self.get_dur(index)
        text = phones
        assert pitch.size(-1) == mel.size(-1)

        # No higher formants?
        if len(pitch.size()) == 1:
            pitch = pitch[None, :]

        # this is a batch
        # FastPitch 1.0: (text, mel, len_text, dur, pitch, speaker)
        return (text, mel, len(text), pitch, energy, speaker, dur,
                audiopath, phones)

    def __len__(self):
        return len(self.audiopaths_and_text)

    @lru_cache()
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

    @lru_cache()
    def get_text(self, text):
        text, text_clean, text_arpabet = self.tp.encode_text(text, return_all=True)
        space = [self.tp.encode_text("A A")[1]]

        if self.prepend_space_to_text:
            text = space + text

        if self.append_space_to_text:
            text = text + space

        return torch.LongTensor(text), text_arpabet

    @lru_cache()
    def get_dur(self, index):
        audiopath, *fields = self.audiopaths_and_text[index]
        name = Path(audiopath).stem

        # TODO: check what happens here with absolute vs relative paths
        path = Path(self.dataset_path, 'durations') if self.dataset_path else Path(audiopath)
        fname = Path(path, name).with_suffix('.pt')

        if self.dur_tmp_dir is not None:
            cached_durpath = Path(self.dur_tmp_dir, fname)
            cached_phonepath = Path(self.dur_tmp_dir, name + '_phones').with_suffix('.pt')
            if cached_durpath.is_file():
                # assume if one exists the other does too
                return torch.load(cached_durpath), torch.load(cached_phonepath)

        if self.load_durs_from_disk:
            duration_path = fields[1]  # assume durations come after pitch
            # assume phone_path is known from duration_path
            phone_path = Path(Path(duration_path).parent, name + '_phones').with_suffix('.pt')
            return torch.load(duration_path), torch.load(phone_path)

        tgt_path = Path(self.textgrid_path, f'{name}.TextGrid')
        try:
            textgrid = read_textgrid(tgt_path, include_empty_intervals=True)
        except FileNotFoundError:
            print(f'{name}.wav TextGrid missing: {tgt_path}')
            raise
        phones, durs, _, _ = parse_textgrid(textgrid.get_tier_by_name('phones'),
                                            self.sampling_rate,
                                            self.hop_length)
        phones = torch.Tensor(self.tp.arpabet_list_to_sequence(phones))
        check_durations(durs, self.get_mel(audiopath).size(1), name)
        durs = torch.Tensor(durs)

        if self.dur_tmp_dir is not None and not cached_durpath.is_file() and not cached_phonepath.is_file():
            return torch.save(durs, cached_durpath), torch.save(phones, cached_phonepath)

        return durs, phones

    @lru_cache()
    def get_pitch(self, index, mel_len=None):
        audiopath, *fields = self.audiopaths_and_text[index]

        if self.n_speakers > 1:
            spk = int(fields[-1])
        else:
            spk = 0

        if self.load_pitch_from_disk:
            pitchpath = fields[0]
            pitch = torch.load(pitchpath)
            if self.pitch_mean is not None:
                assert self.pitch_std is not None
                pitch = normalize_pitch(pitch, self.pitch_mean, self.pitch_std)
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

        pitch_mel = estimate_pitch(wav, mel_len, self.f0_method,
                                   self.pitch_mean, self.pitch_std)

        if self.pitch_tmp_dir is not None and not cached_fpath.is_file():
            cached_fpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pitch_mel, cached_fpath)

        return pitch_mel


class TTSCollate:
    """Zero-pads model inputs and targets based on number of frames per step"""
    # (text, mel, len(text), pitch, energy, speaker, dur, audiopath, phones) = batch
    # 0: text
    # 1: mel
    # 2: len_text
    # 3: pitch
    # 4: energy
    # 5: speaker
    # 6: dur
    # 7: audiopath
    # 8: phones
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

        dur_padded = torch.zeros_like(text_padded, dtype=torch.int32)

        dur_lens = torch.zeros(dur_padded.size(0), dtype=torch.int32)
        for i in range(len(ids_sorted_decreasing)):
            dur = batch[ids_sorted_decreasing[i]][6]
            # With MFA durations:
            # some mismatch between phones in transcript vs phones from text preprocessing
            # for now using phones from texgrid as input
            # PREP DATASET: DUR = LIST, TRAIN: DUR = TENSOR
            dur_padded[i, :len(dur)] = dur
            dur_lens[i] = len(dur)
            assert dur_lens[i] == input_lengths[i]

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
        phones_padded = torch.zeros_like(text_padded, dtype=int)
        for i in range(len(ids_sorted_decreasing)):
            pitch = batch[ids_sorted_decreasing[i]][3]
            energy = batch[ids_sorted_decreasing[i]][4]
            phones = batch[ids_sorted_decreasing[i]][8]
            pitch_padded[i, :, :pitch.shape[1]] = pitch
            energy_padded[i, :energy.shape[0]] = energy
            phones_padded[i, :phones.shape[0]] = phones

        if batch[0][5] is not None:
            speaker = torch.zeros_like(input_lengths)
            for i in range(len(ids_sorted_decreasing)):
                speaker[i] = batch[ids_sorted_decreasing[i]][5]
        else:
            speaker = None

        # Count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)

        audiopaths = [batch[i][7] for i in ids_sorted_decreasing]

        return (text_padded, dur_padded, input_lengths, mel_padded, output_lengths, len_x,
                pitch_padded, energy_padded, dur_lens, speaker, audiopaths, phones_padded)


def batch_to_gpu(batch):
    (text_padded, durs_padded, input_lengths, mel_padded, output_lengths, len_x,
     pitch_padded, energy_padded, dur_lens, speaker, audiopaths, phones_padded) = batch

    text_padded = to_gpu(text_padded).long()
    durs_padded = to_gpu(durs_padded).long()
    dur_lens = to_gpu(dur_lens).long()
    input_lengths = to_gpu(input_lengths).long()
    mel_padded = to_gpu(mel_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    pitch_padded = to_gpu(pitch_padded).float()
    energy_padded = to_gpu(energy_padded).float()
    phones_padded = to_gpu(phones_padded).long()
    if speaker is not None:
        speaker = to_gpu(speaker).long()

    # Alignments act as both inputs and targets - pass shallow copies
    x = [text_padded, input_lengths, mel_padded, output_lengths,
         pitch_padded, energy_padded, speaker, durs_padded, audiopaths, phones_padded]
    y = [mel_padded, durs_padded, dur_lens, output_lengths]
    len_x = torch.sum(output_lengths)
    return (x, y, len_x)
