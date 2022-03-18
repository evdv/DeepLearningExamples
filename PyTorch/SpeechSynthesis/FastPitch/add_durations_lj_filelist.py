import os
from pathlib import Path


def add_duration_column(filename, output_filename):
    all_info = []
    with open(filename) as f:
        for line in f:
            file_path, pitch_path, transcript = line.strip().split('|', maxsplit=2)
            name_stem = Path(os.path.basename(file_path)).stem
            # stop hard-coding which columns already exist (no mels or speakers)
            all_info.append('|'.join([file_path,
                                      pitch_path,
                                      f'durations/{name_stem}.pt',
                                      transcript]))

    with open(output_filename, 'w') as f:
        f.writelines('\n'.join(all_info))


if __name__ == '__main__':
    filelists = {'filelists/ljs_audio_pitch_text_test.txt': 'filelists/ljs_audio_pitch_durs_text_test.txt',
                 'filelists/ljs_audio_pitch_text_train_v3.txt': 'filelists/ljs_audio_pitch_durs_text_train_v3.txt',
                 'filelists/ljs_audio_pitch_text_val.txt': 'filelists/ljs_audio_pitch_durs_text_val.txt'}
    for file_name, output_name in filelists.items():
        add_duration_column(file_name, output_name)
