from utils.hparams import HParams
from utils.audio_augmentation import AudioAugmentation
from utils.feature_extraction import FeatureExtractionCNN, SongExtractionCNN, FeatureExtractionSVM, SongExtractionSVM
import utils.data_manager
from model.cnn import CNN

import os
import torch
from torch import optim
import numpy as np
import pickle
import pandas
import librosa
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

class Recommender:
    def get_recommendation(self, song):
        param = HParams()
    
        # CNN

         # Train the model
        if param.train:
            # Use data augmentation
            if param.data_augmentation:
                AudioAugmentation(param)

            # Extract features
            if param.feature_extraction:
                FeatureExtractionCNN(param)

            train_loader, valid_loader, test_loader = utils.data_manager.get_dataloader(param)
            model = CNN()
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=param.learning_rate, momentum=param.momentum, weight_decay=1e-6, nesterov=True)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=param.factor, patience=param.patience, verbose=True)
            learning_rate = param.learning_rate
            stopping_rate = param.stopping_rate

            print('Training' + param.classifier)
            for epoch in range(param.num_epochs):
                train_loss, train_acc = train_epoch(train_loader,model, optimizer, criterion, param)
                valid_loss, valid_acc = eval_epoch(valid_loader, model, criterion, param)

                print("[Epoch %d/%d] [Train Loss: %.4f] [Train Acc: %.4f] [Valid Loss: %.4f] [Valid Acc: %.4f]" %
                    (epoch + 1, param.num_epochs, train_loss, train_acc, valid_loss, valid_acc))

                if early_stop(scheduler, learning_rate, optimizer, stopping_rate, valid_loss, epoch+1):
                    break

            test_loss, test_acc = eval_epoch(test_loader, model, criterion, param)
            print("Training Finished")
            print("Test Accuracy: %.2f%%" % (100*test_acc))
            torch.save(model, os.path.join('./model', 'CNN.pt'))
        
        # Load the model
        else:
            model = torch.load(os.path.join('./model', 'CNN.pt'))

        # Recomendation
        if param.recommend:
            # Pass the possible recomoended songs throw the network and save their computed genre
            if param.process_recommendations:
                print('Processing recomendations')
                genres = dict()
                for f in os.listdir(param.recommend_path):
                    print(f)
                    y, sr = librosa.load(os.path.join(param.recommend_path, f), param.sample_rate, offset=90, duration=30)
                    feat = SongExtractionCNN(y, param).get_feature()
                    genre = compute_genre(feat, model)
                    genres.update({f:genre})
                with open('CNN_ouput.p','wb') as fp:
                    pickle.dump(genres, fp, protocol=pickle.HIGHEST_PROTOCOL)
                print('Recomendations information saved')
            # Load genres of possible recommended songs
            else:
                with open('CNN_ouput.p','rb') as fp:
                    genres = pickle.load(fp)
            #song, sr = librosa.load(os.path.join('./dataset', param.song), param.sample_rate, offset=90, duration=30)
            feat = SongExtractionCNN(song, param).get_feature()
            genreCNN = compute_genre(feat, model)
            recommendationCNN = compute_recommendation(genreCNN, genres)
            print(recommendationCNN)
           
        # SVM

        if param.train:
            if param.feature_extraction:
                FeatureExtractionSVM(param)
            
            data_set=pandas.read_csv(os.path.join(param.feat_path, 'data_set.csv'),index_col=False)
            number_of_rows,number_of_cols = data_set.shape
            data_set_values = np.array(data_set)
            train, test = train_test_split(data_set_values, test_size = 0.85,random_state=2,
                                           stratify=data_set_values[:,number_of_cols-1])
            train_x=train[:,:number_of_cols-1]
            train_y=train[:,number_of_cols-1]

            test_x=test[:,:number_of_cols-1]
            test_y=test[:,number_of_cols-1]

            print("Training data size: {}".format(train.shape))
            print("Test data size: {}".format(test.shape))
            svm=SVC(C=100,gamma=0.08,probability=True)
            svm.fit(train_x,train_y)
            print("Training Score: {:.3f}".format(svm.score(train_x,train_y)))
            print("Test score: {:.3f}".format(svm.score(test_x,test_y)))
            pickle.dump(svm, open(os.path.join('./model', 'SVM'), 'wb'))
        else:
            svm=pickle.load(open(os.path.join('./model', 'SVM'), 'rb'))

        # Recomendation
        if param.recommend:
            # Pass the possible recomoended songs throw the network and save their computed genre
            if param.process_recommendations:
                print('Processing recomendations')
                genres = dict()
                for f in os.listdir(param.recommend_path):
                    print(f)
                    y, sr = librosa.load(os.path.join(param.recommend_path, f), param.sample_rate, offset=90, duration=30)
                    feat = SongExtractionSVM(y, param).get_feature()
                    genre = svm.predict_proba(feat)
                    genres.update({f:genre})
                with open('SVM_ouput.p','wb') as fp:
                    pickle.dump(genres, fp, protocol=pickle.HIGHEST_PROTOCOL)
                print('Recomendations information saved')
            else:
                with open('SVM_ouput.p','rb') as fp:
                    genres = pickle.load(fp)
            #song, sr = librosa.load(os.path.join('./dataset', param.song), param.sample_rate, offset=90, duration=30)
            feat = SongExtractionSVM(song, param).get_feature()
            genreSVM = svm.predict_proba(feat)
            recommendationSVM = compute_recommendation(genreSVM, genres)
            print(recommendationSVM)
            
        return genreCNN, genreSVM, recommendationCNN, recommendationSVM
                

def accuracy(source, target):
    source = source.max(1)[1].long().cpu()
    target = target.long().cpu()
    correct = (source == target).sum().item()

    return correct/float(source.size(0))

def train_epoch(dataloader, model, optimizer, criterion, hparams):
    model.train()
    
    epoch_loss = 0
    epoch_acc = 0

    for batch, (x, y) in enumerate(dataloader):
        x = x.to(hparams.device)
        y = y.to(hparams.device)

        prediction = model(x)
        loss = criterion(prediction, y.long())
        acc = accuracy(prediction, y.long())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += prediction.size(0)*loss.item()
        epoch_acc += prediction.size(0)*acc

    epoch_loss = epoch_loss/len(dataloader.dataset)
    epoch_acc = epoch_acc/len(dataloader.dataset)

    return epoch_loss, epoch_acc

def eval_epoch(dataloader, model, criterion, hparams):
    model.eval()
  
    epoch_loss = 0
    epoch_acc = 0
    
    for batch, (x, y) in enumerate(dataloader):
        x = x.to(hparams.device)
        y = y.to(hparams.device)

        prediction = model(x)
        loss = criterion(prediction, y.long())
        acc = accuracy(prediction, y.long())

        epoch_loss += prediction.size(0)*loss.item()
        epoch_acc += prediction.size(0)*acc

    epoch_loss = epoch_loss/len(dataloader.dataset)
    epoch_acc = epoch_acc/len(dataloader.dataset)

    return epoch_loss, epoch_acc

def early_stop(scheduler, learning_rate, optimizer, stopping_rate, loss, epoch):
    scheduler.step(loss, epoch)
    learning_rate = optimizer.param_groups[0]['lr']
    stop = learning_rate < stopping_rate

    return stop

def compute_genre(song, model):
    model.eval()
    out = np.zeros(10)
    for piece in song:
        with torch.no_grad():
            x = torch.Tensor(piece).unsqueeze(0).to('cpu')
            y = model(x)
            out = out + softmax(y.squeeze().data.numpy())
    return out/len(song)

def compute_recommendation(genre, songs_recommendations):
    similarity = list()
    print(genre)
    for name, genre_recommendation in songs_recommendations.items():
        dist = np.linalg.norm(softmax(genre)-softmax(genre_recommendation))
        similarity.append((name,dist))
    similarity.sort(key=lambda x: x[1])
    recommendation = list()
    for i in range(5):
        recommendation.append(similarity[i][0])
    return recommendation
