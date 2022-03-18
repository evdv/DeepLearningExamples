import argparse
import os
import pathlib

from common.utils import load_filepaths_and_text


def create_lab_files(dataset_path, filelist, n_speakers):
    # Expect a list of filenames
    if type(filelist) is str:
        filelist = [filelist]

    # difficulty: dealing with 'are there speaker codes are not'?
    dataset_entries = load_filepaths_and_text(filelist, dataset_path,
                                              (n_speakers > 1))

    for filepath, text in dataset_entries:
        wav_name = pathlib.Path(filepath).stem
        # lab extension is hardcoded
        # so is the use of the wavs subdirectory
        lab_filepath = os.path.join(dataset_path, f'{wav_name}.lab')
        with open(lab_filepath, 'w') as f:
            f.write(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--filelist', type=str, required=True, nargs='+',
                        help='List of wavs with transcript')
    parser.add_argument('--n-speakers', type=int, default=1,
                        help='Number of speakers in dataset')
    args = parser.parse_args()

    create_lab_files(args.dataset, args.filelist, args.n_speakers)
