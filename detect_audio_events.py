import argparse
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Tuple, List
from scipy import signal
from sklearn.cluster import KMeans


def main(args):
    y, sampling_rate = load_audio(args.input_audio)
    timepoints, features = get_spectrogram_info(
        y, sampling_rate, args.cut_freq, args.show_spectrogram)
    events_indx = cluster_events(features)
    segments = get_time_segments(events_indx, timepoints)
    format_and_save_results(segments, args.output)


def load_audio(path:str) -> Tuple[np.array, int]:
    y, sampling_rate = librosa.load(path, sr=None)
    print(f'File length {librosa.get_duration(y, sampling_rate)} seconds')
    return y, sampling_rate


def get_spectrogram_info(y: np.array, sampling_rate:int, cut_freq:int,
                         show_spectrogram:bool) -> Tuple[np.array, np.array]:

    f, t, Sxx = signal.spectrogram(y, sampling_rate, nfft=1028, scaling='spectrum')
    if show_spectrogram:
        plt.pcolormesh(t, f[np.where(f < cut_freq)], Sxx[np.where(f < cut_freq)])
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
    data = Sxx[np.where(f<cut_freq)].T
    print(f'Shape of extracted features data {data.shape}')
    return t, data


def cluster_events(features: np.array) -> np.array:
    kmeans = KMeans(n_clusters=2, random_state=0).fit_predict(features)
    return np.where(kmeans==0)[0]


def get_time_segments(events_indx:np.array, timepoints: np.array) -> List[List[float]]:
    segments = [[]]
    for indx, tp in enumerate(timepoints):
        if indx in events_indx:
            segments[-1].append(tp)
        else:
            segments.append([tp])
    return segments


def format_and_save_results(segments: List[List[float]], output_path: str):
    results = pd.DataFrame(columns=['duration', 'timestamp'])
    for i, segment in enumerate(segments):
        results = results.append({
            'duration': max(segment) - min(segment),
            'timestamp': min(segment)
        }, ignore_index=True)
    results = results[results.duration>0]
    results['rank'] = results['duration'].rank(ascending=False)
    results.to_csv(output_path, index=False)
    print(f'Results ar saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-audio', required=True, type=str, help='path to the audio file')
    parser.add_argument('--cut-freq', required=False, default=5000, type=int,
                        help='high limit frequency to consider for feature extraction')
    parser.add_argument('--show-spectrogram', action=argparse.BooleanOptionalAction, default=False, required=False,
                        help='if to show the spectrogram of the audio signal')
    parser.add_argument('--output', required=False, default='./results_of_audio_event_detection.csv', type=str,
                        help='the path for the result csv file')
    args = parser.parse_args()
    main(args)