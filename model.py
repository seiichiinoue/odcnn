import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from music_processor import *

"""
On the paper,
Starting from a stack of three spectrogram excerpts,
convolution and max-pooling in turns compute a set of 20 feature maps 
classified with a fully-connected network.
"""

class convNet(nn.Module):
    """
    copies the neural net used in a paper.
    "Improved musical onset detection with Convolutional Neural Networks".
    src: https://ieeexplore.ieee.org/document/6854953
    """

    def __init__(self):

        super(convNet, self).__init__()
        # model
        self.conv1 = nn.Conv2d(3, 10, (3, 7))
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(1120, 256)
        self.fc2 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120, 1)


    def forward(self, x, istraining=False, minibatch=1):

        x = F.max_pool2d(F.relu(self.conv1(x)), (3, 1))
        x = F.max_pool2d(F.relu(self.conv2(x)), (3, 1))
        x = F.dropout(x.view(minibatch, -1), training=istraining)
        x = F.dropout(F.relu(self.fc1(x)), training=istraining)
        x = F.dropout(F.relu(self.fc2(x)), training=istraining)

        return F.sigmoid(self.fc3(x))


    def train_data_builder(self, feats, answer, major_note_index, samplerate, soundlen=15, minibatch=1, split=0.2):
        """
        Args:
            feats: song.feats; Audio module
            answers: song.answers; Audio module
            major_note_index: answer labels; corresponding to feats
            samplerate: song.samplerate; Audio module
            soundlen: =15. 学習モデルに渡す画像データの横方向の長さ．ここでは(80 * 15)サイズのデータを使用している
            minibatch: training minibatch
            split: =1. 
        Variables:
            minspace: minimum space between major note indexs
            maxspace: maximum space between major note indexs
            idx: index of major_note_index or feats
            dist: distance of two notes
        """

        # acceptable interval in seconds
        minspace = 0.1
        maxspace = 0.7

        idx = np.random.permutation(major_note_index.shape[0] - soundlen) + soundlen // 2
        X, y = [], []
        cnt = 0
        
        for i in range(int(idx.shape[0] * split)):

            dist = major_note_index[idx[i] + 1] - major_note_index[idx[i]]  # distinguish by this value
            
            if dist < maxspace * samplerate / 512 and dist > minspace * samplerate / 512:    
                for j in range(-1, dist + 2):
                    X.append(feats[:, :, major_note_index[idx[i]] - soundlen // 2 + j : major_note_index[idx[i]] + soundlen // 2 + j + 1])
                    y.append(answer[major_note_index[idx[i]] + j])
                    cnt += 1
                    
                    if cnt % minibatch == 0:
                        yield (torch.from_numpy(np.array(X)).float(), torch.from_numpy(np.array(y)).float())
                        X, y = [], []


    def infer_data_builder(self, feats, soundlen=15, minibatch=1):
        
        x = []
        
        for i in range(feats.shape[2] - soundlen):
            x.append(feats[:, :, i:i+soundlen])
        
            if (i + 1) % minibatch == 0:
                yield (torch.from_numpy(np.array(x)).float())
                x = []
        
        if len(x) != 0:
            yield (torch.from_numpy(np.array(x)).float())


    def train(self, songs, minibatch, epoch, device, soundlen=15, val_song=None, save_place='./models/pytorch/model.pth', log='./log/pytorch/log.txt', don_ka=0):
        """
        Args:
            songs: the list of song
            minibatch: minibatch value
            epoch: number of train 
            device: cpu / gpu
            soundlen: width of one train data's image
            val_song: validation song, if you wanna validation while training, give a path of validation song data.
            save_place: save place path
            log: log place path
            don-ka: don(1) or ka(2) or both(0), usually, firstly, train don, then, train ka.
        """

        for song in songs:
            
            timing = song.timestamp[:, 0]
            sound  = song.timestamp[:, 1]
            song.answer = np.zeros((song.feats.shape[2]))

            if don_ka == 0:
                song.major_note_index = np.rint(timing[np.where(sound != 0)] * song.samplerate/512).astype(np.int32)
            else:
                song.major_note_index = np.rint(timing[np.where(sound == don_ka)] * song.samplerate/512).astype(np.int32)
                song.minor_note_index = np.rint(timing[np.where(sound == 3-don_ka)] * song.samplerate/512).astype(np.int32)
            
            song.major_note_index = np.delete(song.major_note_index, np.where(song.major_note_index >= song.feats.shape[2]))
            song.minor_note_index = np.delete(song.minor_note_index, np.where(song.minor_note_index >= song.feats.shape[2]))
            song.answer[song.major_note_index] = 1
            song.answer[song.minor_note_index] = 0.26
            song.answer = milden(song.answer)

        if val_song:

            timing = val_song.timestamp[:, 0]
            sound = val_song.timestamp[:, 1]
            val_song.answer = np.zeros((val_song.feats.shape[2]))

            if don_ka == 0:
                val_song.major_note_index = np.rint(timing[np.where(sound != 0)] * val_song.samplerate/512).astype(np.int32)
            else:
                val_song.major_note_index = np.rint(timing[np.where(sound == don_ka)] * val_song.samplerate/512).astype(np.int32)
                val_song.minor_note_index = np.rint(timing[np.where(sound == 3-don_ka)] * val_song.samplerate/512).astype(np.int32)

            val_song.major_note_index = np.delete(val_song.major_note_index, np.where(val_song.major_note_index >= val_song.feats.shape[2]))
            val_song.minor_note_index = np.delete(val_song.minor_note_index, np.where(val_song.minor_note_index >= val_song.feats.shape[2]))
            val_song.answer[val_song.major_note_index] = 1
            val_song.answer[val_song.minor_note_index] = 0.26
            val_song.answer = milden(val_song.answer)

        # training
        optimizer = optim.SGD(self.parameters(), lr=0.02)
        criterion = nn.MSELoss()
        running_loss = 0
        val_loss = 0

        for i in range(epoch):
            for song in songs:
                for X, y in self.train_data_builder(song.feats, song.answer, song.major_note_index, song.samplerate, soundlen, minibatch, split=0.2):
                    optimizer.zero_grad()
                    output = self(X.to(device), istraining=True, minibatch=minibatch)
                    target = y.to(device)
                    loss = criterion(output.squeeze(), target)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.data.item()

            with open(log, 'a') as f:
                print("epoch: %.d running_loss: %.10f " % (i+1, running_loss), file=f)

            print("epoch: %.d running_loss: %.10f" % (i+1, running_loss))
            
            running_loss = 0    

            if val_song:
                inference = torch.from_numpy(self.infer(val_song.feats, device, minibatch=512)).to(device)
                target = torch.from_numpy(val_song.answer[:-soundlen]).float().to(device)
                loss = criterion(inference.squeeze(), target)
                val_loss = loss.data.item()

                with open(log, 'a') as f:
                    print("val_loss: %.10f " % (val_loss), file=f)

        torch.save(self.state_dict(), save_place)


    def infer(self, feats, device, minibatch=1):

        with torch.no_grad():
            inference = None
            for x in tqdm(self.infer_data_builder(feats, minibatch=minibatch), total=feats.shape[2]//minibatch):
                output = self(x.to(device), minibatch=x.shape[0])
                if inference is not None:
                    inference = np.concatenate((inference, output.cpu().numpy().reshape(-1)))
                else:
                    inference = output.cpu().numpy().reshape(-1)
            
            return np.array(inference).reshape(-1)


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = convNet()
    net = net.to(device)
           
    with open('./data/pickles/train_data.pickle', mode='rb') as f:
        songs = pickle.load(f)

    minibatch = 128
    soundlen = 15
    epoch = 100

    if sys.argv[1] == 'don':
        net.train(songs=songs, minibatch=minibatch, val_song=None, epoch=epoch, device=device, soundlen=soundlen, save_place='./models/pytorch/don_model.pth', log='./data/log/pytorch/don.txt', don_ka=1)
    
    if sys.argv[1] == 'ka':
        net.train(songs=songs, minibatch=minibatch, val_song=None, epoch=epoch, device=device, soundlen=soundlen, save_place='./models/pytorch/ka_model.pth', log='./data/log/pytorch/ka.txt', don_ka=2)

