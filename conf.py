
threads_used = 6

#####################################
# data paths                        #
#####################################

dir_path = 'C:/Users/Christian/kaggle_data/'

# folders to training and testing files
train_data = 'FSDKaggle2018.audio_train/'
test_data = 'FSDKaggle2018.audio_test/'

# paths to meta files of dataset
meta_train = dir_path + 'FSDKaggle2018.meta/' + 'train_post_competition.csv'
meta_test = dir_path + 'FSDKaggle2018.meta/' + 'test_post_competition_scoring_clips.csv'

# collected data output
train_np_file = 'x_train.npy'
test_np_file = 'x_test.npy'


#####################################
# spectrogram constants             #
#####################################

load_duration = 3
sample_rate = 44100
fft_window_size = 2024
hop_length = 512


#####################################
# model parameter                   #
#####################################

epochs = 5
batch_size = 16
model_file_name = 'model_{time}_{epoch}e.h5'.format(epoch=epochs, time='{time}')
encoder_output_layer = 16


#####################################
# search parameter                  #
#####################################

model_path = '/models/model_151930_30e.h5'
space_file = 'search_space.npy'
space_meta_file = 'space_meta.json'
tsne_space = 'tsne_data.npy'
pca_space = 'pca_data.npy'

labels = ["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime",
          "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano",
          "Fart", "Finger_snapping", "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica",
          "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors",
          "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle",
          "Writing"]

