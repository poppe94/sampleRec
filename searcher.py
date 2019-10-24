import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import collections
import json
from keras import Model
import seaborn as sns

import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial import cKDTree

from gatherData import DataCollector
from model import Autoencoder
import conf


class Searcher:
    """This class wraps the searcher, creating a searchespace and giving recommendations.
    """
    def __init__(self):
        self.data = []
        self.space_data = []
        self.space_meta = []
        self.files = []

        self.collector = DataCollector(conf.dir_path)

        self.ae = Autoencoder()
        self.ae.load_model(conf.model_path)
        self.encoder = Model(self.ae.model.input, self.ae.model.layers[conf.encoder_output_layer].output)

    def get_feature_vector(self, spec_data):
        """
        This generates a feature vector from provided spectrogram data using the trained encoder model.
        :param spec_data: spectrogram data
        :return: A feature vector with 1024 dimensions.
        """
        spec_data = (librosa.power_to_db(spec_data, ref=np.max) + 80) / 80
        spec_data = np.reshape(spec_data, (1,) + self.ae.shape)
        return self.encoder.predict(spec_data)

    def draw_features(self, spec_data, recon_data, feature_vector):
        """Draw data in parameters.
        :param spec_data:
        :param recon_data:
        :param feature_vector:
        """
        if spec_data is not None:
            plt.figure()
            plt.axis('off')
            librosa.display.specshow(spec_data.reshape(128, 256),
                                     y_axis='mel',
                                     fmax=8000,
                                     x_axis='time')
            plt.show()
        if recon_data is not None:
            plt.figure()
            plt.axis('off')
            back_scaled = recon_data * 80 - 80
            librosa.display.specshow(back_scaled.reshape(128, 256),
                                     y_axis='mel',
                                     fmax=8000,
                                     x_axis='time')
            plt.show()
        if feature_vector is not None:
            plt.figure()
            plt.axis('off')
            plt.imshow(feature_vector.reshape(16, 8 * 8))
            plt.show()

    def create_search_space(self):
        """
        This creates the search space for the recommendation. Uses data from DataCollector class.
        From this data Feature vectors are computed. These are also saved on disk.
        """
        self.data = self.collector.collect_training_data()

        result = collections.defaultdict(list)
        for d in self.data:
            result[d['filename']].append(d)
        result_list = list(result.values())

        for entry in result_list:
            d = entry[0]
            feature_vectors = []
            if len(entry) > 1:
                for part in entry:
                    vector = self.get_feature_vector(part['data'])
                    feature_vectors.append(vector)
                vec = np.average(np.array(feature_vectors), axis=0)[0]
            else:
                vec = self.get_feature_vector(entry[0]['data'])
            vec = np.reshape(vec, (8, 16, 8))
            d['data'] = len(self.space_data)
            self.space_data.append(vec / np.max(vec))
            if d['filename'] not in self.files:
                self.files.append(d['filename'])
            self.space_meta.append(d)
        self.space_data = np.asarray(self.space_data)
        print('Writing search space files.')
        np.save(conf.dir_path + conf.space_file, self.space_data)
        with open(conf.dir_path + conf.space_meta_file, 'w') as f:
            json.dump(self.space_meta, f)

    def load_space(self):
        """
        Loads a saved feature vector file.
        """
        try:
            self.space_data = np.load(conf.dir_path + conf.space_file)
            with open(conf.dir_path + conf.space_meta_file, 'r') as f:
                self.space_meta = json.load(f)
        except FileNotFoundError:
            print('Required files not found, running data operations.')
            self.create_search_space()

    def compute_tsne(self, draw=False):
        """
        Creates a 2D feature space and cann plot an image of this space.
        :param draw: Draws pyplot when true.
        """
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

        shape = self.space_data.shape[1] * self.space_data.shape[2] * self.space_data.shape[3]
        data_reshaped = np.reshape(self.space_data, (self.space_data.shape[0], shape))
        feat_cols = ['value'+str(i) for i in range(data_reshaped.shape[1])]
        df = pd.DataFrame(data_reshaped, columns=feat_cols)
        df['label'] = [d['label'] for d in self.space_meta]

        print('Fitting tsne.')
        tsne_results = tsne.fit_transform(df[feat_cols].values)

        np.save(conf.dir_path + conf.tsne_space, tsne_results)

        if draw:
            df['tsne-2d-one'] = tsne_results[:, 0]
            df['tsne-2d-two'] = tsne_results[:, 1]

            plt.figure(figsize=(16, 10))
            sns.scatterplot(
                x='tsne-2d-one', y='tsne-2d-two',
                hue='label',
                palette=sns.color_palette("hls", len(conf.labels)),
                data=df,
                legend="full",
                alpha=0.3
            )
            plt.show()

    def compute_pca(self, draw=False):
        """
        Creates a 2D feature space and can plot an image of this space.
        :param draw: Draws pyplot when true.
        """
        pca = PCA(n_components=2)

        shape = self.space_data.shape[1] * self.space_data.shape[2] * self.space_data.shape[3]
        data_reshaped = np.reshape(self.space_data, (self.space_data.shape[0], shape))
        feat_cols = ['value' + str(i) for i in range(data_reshaped.shape[1])]
        df = pd.DataFrame(data_reshaped, columns=feat_cols)
        df['label'] = [d['label'] for d in self.space_meta]

        pca_result = pca.fit_transform(df[feat_cols].values)
        np.save(conf.dir_path + conf.pca_space, pca_result)

        if draw:
            df['pca-one'] = pca_result[:, 0]
            df['pca-two'] = pca_result[:, 1]

            plt.figure(figsize=(16, 10))
            sns.scatterplot(
                x="pca-one", y="pca-two",
                hue="label",
                palette=sns.color_palette("hls", len(conf.labels)),
                data=df,
                legend="full",
                alpha=0.3
            )
            plt.show()

    def get_recommendation(self, filename):
        """
        This searches the recommended audio files for the provided audio sample. From the input a spectrogram is
        extracted and the feature vector is computed. A kd-tree is utilized to find five neighbours to the feature
        vector of the input.
        :param filename: Audio sample to find recommendations for.
        :return: The filenames of the five closest samples.
        """
        results = []
        splits = []
        feature_vectors = []

        file_path = conf.dir_path + filename
        amount_of_samples = conf.load_duration * conf.sample_rate
        audio, rate = librosa.load(file_path, sr=conf.sample_rate, res_type='kaiser_fast')
        if audio.size < amount_of_samples:
            print('File is to small.')
            return
        for i in range(0, audio.size, amount_of_samples):
            splits.append(audio[i:i + amount_of_samples])
        if splits[-1].size < amount_of_samples:
            del splits[-1]
        for s in splits:
            if s.size < amount_of_samples:
                continue
            data = self.collector.extract_mel_spectrogram(s)
            data = data[:, :-3]
            results.append(data)

        if len(results) > 1:
            for part in results:
                vector = self.get_feature_vector(part)
                feature_vectors.append(vector)
            vec = np.average(np.array(feature_vectors), axis=0)[0]
        else:
            vec = self.get_feature_vector(results[0])
        vec = np.reshape(vec, 8 * 16 * 8)
        vec = vec / np.max(vec)
        # self.draw_features(None, None, vec)
        pca = PCA(n_components=64)

        shape = self.space_data.shape[1] * self.space_data.shape[2] * self.space_data.shape[3]
        data_reshaped = np.reshape(self.space_data, (self.space_data.shape[0], shape))
        data_reshaped = np.append(data_reshaped, [vec], axis=0)

        pca_result = pca.fit_transform(data_reshaped)
        input = pca_result[-1]
        pca_result = pca_result[:-1]
        kd_tree = cKDTree(pca_result, leafsize=100)

        d, index = kd_tree.query(input, k=5, distance_upper_bound=6)
        print('Input was ' + filename)
        print('Your recommended files are:')
        for i in index:
            print(self.space_meta[i]['filename'])
        return

