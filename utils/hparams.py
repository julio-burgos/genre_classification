import torch

class HParams(object):
    def __init__(self):
        # Program parameters
              
        self.classifier = 'CNN'
        self.train = False
        self.data_augmentation = False
        self.feature_extraction = False

        self.recommend = True
        self.process_recommendations = False
        self.play_recomendation = False

        # Directories
        self.dataset_path = './dataset/gtzan'
        self.mel_path= './melspectogram'
        self.feat_path = './dataset/features'
        self.recommend_path = './dataset/recommendation'
        self.genres =  ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae','rock']
        self.song = 'rosalia.fcking_money_man_milionaria_dio_no_libre_del_dinero.mp3'
        
        # Feature Parameters
        self.sample_rate=22050
        self.fft_size = 1024
        self.win_size = 1024
        self.hop_size = 512
        self.num_mels = 128
        self.feature_length = 1024
        self.frame_size = 2048

        # Training Parameters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 16
        self.num_epochs = 50
        self.learning_rate = 1e-2
        self.stopping_rate = 1e-5
        self.weight_decay = 1e-6
        self.momentum = 0.9
        self.factor = 0.2
        self.patience = 5
