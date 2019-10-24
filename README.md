# sampleReco

is an audio sample recommender prototype. It utilizes an unsupervised machine learning aproach to learn a 
low dimensional representaion of audio data. This representaion is used to construct a search space for recommendation. 

## Install

Follow these instructions to get the project locally. The following software is required.

```
$ pip install tensorflow
```
Or for GPUs that can run CUDA 
```
$ pip install tesorflow-gpu
```
```
$ pip install keras
```
```
$ pip install scikit
```
```
$ pip install librosa
```
To get a live visulization of you training model (optional). For more info see 
[the Tensorboard gitlab page](https://github.com/tensorflow/tensorboard)
```
$ pip install tensorboard 
```
#### For visualization only
```
$ pip install matplotlib 
```
```
$ pip install seaborn
```
Or you can simply use the provided `requirements.txt`
```
$ pip install requirements.txt
```
## Usage
The following instruction will guid you through the data acquisition process, how to train a model and how to use this 
model to get Sample recommendations. 

#### Generate training and testing data

You will need an sufficiently large audio data set to train a model. This prototype was created with the 
[FSDKaggle2018](https://zenodo.org/record/2552860#.XFD05fwo-V4) dataset. I recommend this to begin with.
To setup the prototype the `config.py` file needs to be filled out with the correct directories.
 ```python
dir_path = '<root dir>'

# folders to training and testing files
train_data = 'FSDKaggle2018.audio_train/'
test_data = 'FSDKaggle2018.audio_test/'

# paths to meta files of dataset
meta_train = dir_path + 'FSDKaggle2018.meta/' + 'train_post_competition.csv'
meta_test = dir_path + 'FSDKaggle2018.meta/' + 'test_post_competition_scoring_clips.csv'

# collected data output
train_np_file = 'x_train.npy'
test_np_file = 'x_test.npy'
```
Now use the `DataCollector` class to extract spectrograms from your audio files. This will also generate the correct
data packages used by the model, these are saved in .npy files. This process utilizes multiple threads `conf.threads_used` defines the amount of 
threads used.
```Python
import conf
from gatherData import DataCollector
collector = DataCollector(conf.dir_path)
collector.collect_data()
collector.save_data()
```
#### Training a model
These statements will start the training of the provided model architecture. It will also trigger the data collection
process to generate the needed .npy files. 
```python
from model import Autoencoder
autoenc = Autoencoder()
autoenc.train_model()
```
Training parameters like batch size and epochs are defined in `conf.py`. To add a completely new model architecture you 
will need to define a new method in the `model.Autencoder` class like for the other existing architectures. This new method
needs to be called in `model.Autoencoder.compile_architecture` method.
##### Encoder output layer
Keep in mind to set the correct encoder output layer index to conf.encoder_output_layer.
#### Getting recommended samples 
These statements will generate a search space from the provided trainings data set. Putting in a path to an Audio file
will generate five recommendation samples. These are sourced from the provided dataset. 
```python
from searcher import Searcher
s = Searcher()
s.load_space()
s.get_recommendation('input file path')
```
## Built With
*  [Pytrhon 3.7](http://python.org)
*  [tensorflow](https://www.tensorflow.org/) - An open source machine learning Platform
*  [keras](https://keras.io/) - A neural networks API for Python
*  [scikit](https://scikit-learn.org/stable/) - Provides data mining and analysis tools
*  [librosa](https://librosa.github.io/librosa/index.html#) - A music and audio analysis package

