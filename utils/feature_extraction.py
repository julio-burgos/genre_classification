import os
import numpy as np
import pandas
import sklearn
import librosa

def load_list(list_name, hparams):
    with open(os.path.join(hparams.dataset_path, list_name)) as f:
        file_names = f.read().splitlines()
        return file_names

def melspectrogram_dataset(file_name, hparams):
    y, sr = librosa.load(os.path.join(hparams.dataset_path, file_name), hparams.sample_rate)
    S = librosa.stft(y, n_fft=hparams.fft_size, hop_length=hparams.hop_size, win_length=hparams.win_size)

    mel_basis = librosa.filters.mel(hparams.sample_rate, n_fft=hparams.fft_size, n_mels=hparams.num_mels)
    mel_S = np.dot(mel_basis, np.abs(S))
    mel_S = np.log10(1+10*mel_S)
    mel_S = mel_S.T

    return mel_S

def melspectrogram(y, hparams):
    S = librosa.stft(y, n_fft=hparams.fft_size, hop_length=hparams.hop_size, win_length=hparams.win_size)

    mel_basis = librosa.filters.mel(hparams.sample_rate, n_fft=hparams.fft_size, n_mels=hparams.num_mels)
    mel_S = np.dot(mel_basis, np.abs(S))
    mel_S = np.log10(1+10*mel_S)
    mel_S = mel_S.T

    return mel_S

def resize_array(array, length):
    resize_array = np.zeros((length, array.shape[1]))
    if array.shape[0] >= length:
        resize_array = array[:length]
    else:
        resize_array = np.hstack((array,np.zeros(length-array.shape[0],array.shaoe[1])))
    return resize_array

class FeatureExtractionCNN:
    def __init__(self, hparams):
        print("Extracting Feature")
        list_names = ['train_list.txt', 'valid_list.txt', 'test_list.txt']

        for list_name in list_names:
            set_name = list_name.replace('_list.txt', '')
            file_names = load_list(list_name, hparams)

            for file_name in file_names:
                feature = melspectrogram_dataset(file_name, hparams)
                feature = resize_array(feature, hparams.feature_length)

                # Data Arguments
                num_chunks = feature.shape[0]/hparams.num_mels
                data_chuncks = np.split(feature, num_chunks)

                for idx, i in enumerate(data_chuncks):
                    save_path = os.path.join(hparams.mel_path, set_name, file_name.split('/')[0])
                    save_name = file_name.split('/')[1].split('.au')[0]+str(idx)+".npy"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    np.save(os.path.join(save_path, save_name), i.astype(np.float32))
                    print(os.path.join(save_path, save_name))

        print('finished')

class SongExtractionCNN:
    def __init__(self, song, hparams):
        print("Extracting Song Features")
        feature = melspectrogram(song, hparams)
        feature = resize_array(feature, hparams.feature_length)

        # Data Arguments
        num_chunks = feature.shape[0]/hparams.num_mels
        self.data_chuncks = np.split(feature, num_chunks)

    def get_feature(self):
        return self.data_chuncks

class FeatureExtractionSVM:
    def __init__(self, hparams):
        samp_rate = hparams.sample_rate
        frame_size = hparams.frame_size
        hop_size = hparams.hop_size
        dataset_dir = hparams.dataset_path
        sub_folders = get_subdirectories(dataset_dir)

        labels = []
        is_created = False

        print("Extracting features from audios...")
        for sub_folder in sub_folders:
            print(".....Working in folder:", sub_folder)
            sample_arrays = get_sample_arrays(dataset_dir, sub_folder, samp_rate)
            for sample_array in sample_arrays:
                row = extract_features(sample_array, samp_rate, frame_size, hop_size)
                if not is_created:
                    dataset_np = np.array(row)
                    is_created = True
                elif is_created:
                    dataset_np = np.vstack((dataset_np, row))

                labels.append(sub_folder)

        print("Normalizing the data...")
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
        dataset_np = scaler.fit_transform(dataset_np)

        Feature_Names = ['meanZCR', 'stdZCR', 'meanSpecCentroid', 'stdSpecCentroid', 'meanSpecContrast', 'stdSpecContrast',
                        'meanSpecBandwidth', 'stdSpecBandwidth', 'meanSpecRollof', 'stdSpecRollof',
                        'meanMFCC_1', 'stdMFCC_1', 'meanMFCC_2', 'stdMFCC_2', 'meanMFCC_3', 'stdMFCC_3',
                        'meanMFCC_4', 'stdMFCC_4', 'meanMFCC_5', 'stdMFCC_5', 'meanMFCC_6', 'stdMFCC_6',
                        'meanMFCC_7', 'stdMFCC_7', 'meanMFCC_8', 'stdMFCC_8', 'meanMFCC_9', 'stdMFCC_9',
                        'meanMFCC_10', 'stdMFCC_10', 'meanMFCC_11', 'stdMFCC_11', 'meanMFCC_12', 'stdMFCC_12',
                        'meanMFCC_13', 'stdMFCC_13'
                        ]
        dataset_pandas = pandas.DataFrame(dataset_np, columns=Feature_Names)

        dataset_pandas["genre"] = labels
        dataset_pandas.to_csv(os.path.join(hparams.feat_path,"data_set.csv"), index=False)
        print("Data set has been created and sent to the project folder!")

class SongExtractionSVM:
    def __init__(self, song, hparams):
        samp_rate = hparams.sample_rate
        frame_size = hparams.frame_size
        hop_size = hparams.hop_size
        dataset_dir = hparams.dataset_path

        row = np.array(extract_features(song, samp_rate, frame_size, hop_size))
        self.features = row.reshape(1, -1)
    
    def get_feature(self):
        return self.features

def get_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def get_sample_arrays(dataset_dir, folder_name, samp_rate):
    path_of_audios = librosa.util.find_files(dataset_dir + "/" + folder_name)
    audios = []
    for audio in path_of_audios:
        x, sr = librosa.load(audio, sr=samp_rate, duration=25.0)
        audios.append(x)
    audios_np = np.array(audios)
    return audios_np


def extract_features(signal, sample_rate, frame_size, hop_size):
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_size, hop_length=hop_size)
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=frame_size,
                                                          hop_length=hop_size)
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sample_rate, n_fft=frame_size,
                                                          hop_length=hop_size)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate, n_fft=frame_size,
                                                            hop_length=hop_size)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)

    return [

        np.mean(zero_crossing_rate),
        np.std(zero_crossing_rate),
        np.mean(spectral_centroid),
        np.std(spectral_centroid),
        np.mean(spectral_contrast),
        np.std(spectral_contrast),
        np.mean(spectral_bandwidth),
        np.std(spectral_bandwidth),
        np.mean(spectral_rolloff),
        np.std(spectral_rolloff),

        np.mean(mfccs[1, :]),
        np.std(mfccs[1, :]),
        np.mean(mfccs[2, :]),
        np.std(mfccs[2, :]),
        np.mean(mfccs[3, :]),
        np.std(mfccs[3, :]),
        np.mean(mfccs[4, :]),
        np.std(mfccs[4, :]),
        np.mean(mfccs[5, :]),
        np.std(mfccs[5, :]),
        np.mean(mfccs[6, :]),
        np.std(mfccs[6, :]),
        np.mean(mfccs[7, :]),
        np.std(mfccs[7, :]),
        np.mean(mfccs[8, :]),
        np.std(mfccs[8, :]),
        np.mean(mfccs[9, :]),
        np.std(mfccs[9, :]),
        np.mean(mfccs[10, :]),
        np.std(mfccs[10, :]),
        np.mean(mfccs[11, :]),
        np.std(mfccs[11, :]),
        np.mean(mfccs[12, :]),
        np.std(mfccs[12, :]),
        np.mean(mfccs[13, :]),
        np.std(mfccs[13, :]),
    ]