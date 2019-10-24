
import csv
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import struct

from threading import Thread, current_thread

import conf


class MyThread(Thread):
    """
    defines the threads for multi threaded file search.
    """
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        # print(self._target)
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


class DataCollector:
    """
    This class handles the data set collection and Feature extraction.
    """
    features_train = []
    features_test = []
    meta_list_train = []
    meta_list_test = []

    def __init__(self, path):
        """
        Initializes the meta files from the dataset.
        :param path: main directory path
        """
        self.path = path
        self.audiodf = None
        with open(conf.meta_train) as f:
            self.meta_list_train = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
        with open(conf.meta_test) as f:
            self.meta_list_test = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]

    @staticmethod
    def chunk_list(seq, num):
        """
        Simply divides a list into n equal chunks.
        :param seq: input list
        :param num: number of parts
        :return: list of chunks from input list
        """
        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            dict_entry = {'start': int(last),
                          'end': int(last + avg),
                          'chunk': seq[int(last):int(last + avg)]
                          }
            out.append(dict_entry)
            last += avg

        return out

    @staticmethod
    def read_file_properties(filename):
        """
        Provides some data analysis for an audio file.
        :param filename: input audio file
        :return: infos about audio file
        """
        wave_file = open(filename, "rb")

        fmt = wave_file.read(36)

        num_channels_string = fmt[10:12]
        num_channels = struct.unpack('<H', num_channels_string)[0]
        sample_rate_string = fmt[12:16]
        sample_rate = struct.unpack("<I", sample_rate_string)[0]
        bit_depth_string = fmt[22:24]
        bit_depth = struct.unpack("<H", bit_depth_string)[0]

        return num_channels, sample_rate, bit_depth

    @staticmethod
    def extract_mel_spectrogram(audio):
        """
        Computes the spectrogram from the provided audio data.
        :param audio: audio data as numpy array
        :return: mel scaled spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(y=audio,
                                                  sr=conf.sample_rate,
                                                  n_fft=conf.fft_window_size,
                                                  hop_length=conf.hop_length,
                                                  window='hann')

        return mel_spec

    @staticmethod
    def plot_spectrum(data, show=False):
        """
        Plots a pectrogram.
        :param data: spectral data
        :param show: shows and saves the plot when true
        """
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.amplitude_to_db(data['data'], ref=np.max),
                                 y_axis='mel',
                                 fmax=8000,
                                 x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('File: {}'.format(data['filename']))
        plt.tight_layout()
        if show:
            plt.show()
            source = data['filename'].split('.')[0]
            filename = conf.dir_path + 'fft{}_hop{}_{}'.format(conf.fft_window_size, conf.hop_length, source)
            plt.savefig(filename)

    @staticmethod
    def plot_mfcc(data):
        """
        Plots and shows mfcc data.
        :param data: mfc coefficients
        """
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(data['data'], x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('File: {}'.format(data['filename']))
        plt.tight_layout()
        plt.show()

    def analyse_file(self):
        """
        Provides som data set analysis.
        """
        audiodata = []
        for row in self.meta_list_train:
            file_path = self.path + 'FSDKaggle2018.audio_train/' + row['fname']
            data = self.read_file_properties(file_path)
            audiodata.append(data)
        # Convert into a Panda dataframe
        self.audiodf = pd.DataFrame(audiodata, columns=['num_channels', 'sample_rate', 'bit_depth'])

    def generate_features_threading(self, input_dict, directory, results):
        """
        Loads the audio files and filters to short files. From longer files multiple spectrograms are generated.
        :param input_dict: list with filenames to compute
        :param directory: directory of files
        :param results: computed data to collect when all threads are finished
        """
        print('Started thread {}.'.format(current_thread().name))
        for row in input_dict['chunk']:
            splits = []
            file_path = self.path + directory + row['fname']
            amount_of_samples = conf.load_duration * conf.sample_rate

            audio, rate = librosa.load(file_path, sr=conf.sample_rate, res_type='kaiser_fast')
            if audio.size < amount_of_samples:
                continue    # discard short files
            for i in range(0, audio.size, amount_of_samples):
                splits.append(audio[i:i + amount_of_samples])
            if splits[-1].size < amount_of_samples:
                del splits[-1]
            for s in splits:
                if s.size < amount_of_samples:
                    continue
                data = self.extract_mel_spectrogram(s)
                data = data[:, :-3]
                results.append({'data': data, 'filename': row['fname'], 'label': row['label']})

            if input_dict['chunk'].index(row) % 20 == 1:
                progress = (input_dict['chunk'].index(row) + 1) / (input_dict['end'] - input_dict['start'])
                print('Progress of {thread_name}:\t {progress:6.2f}% {entry} of {max}'.format(
                    thread_name=current_thread().name,
                    progress=progress*100.0,
                    entry=input_dict['chunk'].index(row),
                    max=input_dict['end'] - input_dict['start']))

            # if input_dict['chunk'].index(row) > 10:
            #     break

    def collect_training_data(self):
        """
        Handles the collection of the training data.
        :return: array with spectrograms of training data
        """
        print('Found {} entries in CSV file.'.format(len(self.meta_list_train)))
        chunks = self.chunk_list(self.meta_list_train, conf.threads_used)
        threads = []
        results = []
        print('Working with {} Threads.'.format(conf.threads_used))
        for chunk in chunks:
            t = MyThread(target=self.generate_features_threading, args=(chunk, conf.train_data, results,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        print('All threads have stopped.')
        return results

    def collect_testing_data(self):
        """
        Handles the collection of the testing data.
        :return: array with spectrograms of testing data
        """
        print('Found {} entries in CSV file.'.format(len(self.meta_list_test)))
        # [{'start': 12, 'end': 58, 'chunk': [<list>]}]
        chunks = self.chunk_list(self.meta_list_test, conf.threads_used)
        # chunks = self.chunk_list(self.meta_list, 1)
        threads = []
        results = []
        print('Working with {} Threads.'.format(conf.threads_used))
        for chunk in chunks:
            t = MyThread(target=self.generate_features_threading, args=(chunk, conf.test_data, results,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        print('All threads have stopped.')
        return results

    def collect_data(self):
        """
        Initiates the collection of training and testing data.
        """
        if not self.meta_list_train:
            raise Exception('Member meta_list is empty.')
        print('Collecting test data.')
        self.features_test = self.collect_testing_data()
        print('Collecting training data.')
        self.features_train = self.collect_training_data()

    def save_data(self):
        """
        Saves the testing and training data arrays to disk.
        """
        if self.features_test:
            x_test = np.array([entry['data'] for entry in self.features_test])
            np.save(conf.dir_path + conf.test_np_file, x_test)
            print('Test data file created.')
        if self.features_train:
            x_train = np.array([entry['data'] for entry in self.features_train])
            np.save(conf.dir_path + conf.train_np_file, x_train)
            print('Train data file created.')


def run_testing():
    """
    This just draws some spectrograms for testing.
    """
    collector = DataCollector(conf.dir_path)
    testing = collector.collect_testing_data()

    plt.title("Mel Spectrogram")
    plt.xlabel("Time in Frames")
    plt.ylabel("Mel-scaled Frequency (Hz)")
    plt.imshow(testing[4]['data'].reshape(128, 256))
    plt.show()

    plt.title("Mel Spectrogram")
    plt.xlabel("Time in Frames")
    plt.ylabel("Mel-scaled Frequency (Hz)")
    plt.imshow(librosa.amplitude_to_db(testing[4]['data'], ref=np.max).reshape(128, 256))
    plt.show()

    plt.title("Mel Spectrogram")
    plt.xlabel("Time in Frames")
    plt.ylabel("Mel-scaled Frequency (Hz)")
    plt.imshow(librosa.power_to_db(testing[4]['data'], ref=np.max).reshape(128, 256))
    plt.show()
