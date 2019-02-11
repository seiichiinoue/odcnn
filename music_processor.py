import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from scipy import signal
from scipy.fftpack import fft
from librosa.filters import mel
from librosa.display import specshow
from librosa import stft
from librosa.effects import pitch_shift
import pickle
import sys
from numba import jit, prange
from sklearn.preprocessing import normalize


class Audio:
    """
    audio class which holds music data and timestamp for notes.

    Args:
        filename: file name.
        stereo: True or False; wether you have Don/Ka streo file or not. normaly True.

    Example:
        >>>from music_processor import *
        >>>song = Audio(filename)
        >>># to get audio data
        >>>song.data
        >>># to import .tja files:
        >>>song.import_tja(filename)
        >>># to get data converted
        >>>song.data = (song.data[:,0]+song.data[:,1])/2
        >>>fft_and_melscale(song, include_zero_cross=False)
    """

    def __init__(self, filename, stereo=True):

        self.data, self.samplerate = sf.read(filename, always_2d=True)
        if stereo is False:
            self.data = (self.data[:, 0]+self.data[:, 1])/2
        self.timestamp = []

    def plotaudio(self, start_t, stop_t):

        plt.plot(np.linspace(start_t, stop_t, stop_t-start_t),
                 self.data[start_t:stop_t, 0])
        plt.show()

    def save(self, filename="./savedmusic.wav", start_t=0, stop_t=None):

        if stop_t is None:
            stop_t = self.data.shape[0]
        sf.write(filename, self.data[start_t:stop_t], self.samplerate)

    def import_tja(self, filename, verbose=False, diff=False, difficulty=None):
        '''imports tja file and convert it into timestamp'''
        
        now = 0.0
        bpm = 100
        measure = [4, 4]  # hyousi
        self.timestamp = []
        skipflag = False

        with open(filename, "rb") as f:
            while True:
                line = f.readline()
                try:
                    line = line.decode('sjis')
                except UnicodeDecodeError:
                    line = line.decode('utf-8')
                if line.find('//') != -1:
                    line = line[:line.find('//')]
                if line[0:5] == "TITLE":
                    if verbose:
                        print("importing: ", line[6:])
                elif line[0:6] == "OFFSET":
                    now = -float(line[7:-2])
                elif line[0:4] == "BPM:":
                    bpm = float(line[4:-2])
                if line[0:6] == "COURSE":
                    if difficulty and difficulty > 0:
                        skipflag = True
                        difficulty -= 1
                elif line == "#START\r\n":
                    if skipflag:
                        skipflag = False
                        continue
                    break
            
            sound = []
            while True:
                line = f.readline()
                # print(line)
                try:
                    line = line.decode('sjis')
                except UnicodeDecodeError:
                    line = line.decode('utf-8')

                if line.find('//') != -1:
                    line = line[:line.find('//')]
                if line[0] <= '9' and line[0] >= '0':
                    if line.find(',') != -1:
                        sound += line[0:line.find(',')]
                        beat = len(sound)
                        for i in range(beat):
                            if diff:
                                if int(sound[i]) in (1, 3, 5, 6, 7):
                                    self.timestamp.append(
                                        [int(100*(now+i*60*measure[0]/bpm/beat))/100, 1])
                                elif int(sound[i]) in (2, 4):
                                    self.timestamp.append(
                                        [int(100*(now+i*60*measure[0]/bpm/beat))/100, 2])
                            else:
                                if int(sound[i]) != 0:
                                    self.timestamp.append(
                                        [int(100*(now+i*60*measure[0]/bpm/beat))/100, int(sound[i])])
                        now += 60/bpm*measure[0]
                        sound = []
                    else:
                        sound += line[0:-2]
                elif line[0] == ',':
                    now += 60/bpm*measure[0]
                elif line[0:10] == '#BPMCHANGE':
                    bpm = float(line[11:-2])
                elif line[0:8] == '#MEASURE':
                    measure[0] = int(line[line.find('/')-1])
                    measure[1] = int(line[line.find('/')+1])
                elif line[0:6] == '#DELAY':
                    now += float(line[7:-2])
                elif line[0:4] == "#END":
                    if(verbose):
                        print("import complete!")
                    self.timestamp = np.array(self.timestamp)
                    break

    def synthesize(self, diff=True, don="./data/don.wav", ka="./data/ka.wav"):
        
        donsound = sf.read(don)[0]
        kasound = sf.read(ka)[0]
        donlen = len(donsound)
        kalen = len(kasound)
        
        if diff is True:
            for stamp in self.timestamp:
                timing = int(stamp[0]*self.samplerate)
                try:
                    if stamp[1] in (1, 3, 5, 6, 7):
                        self.data[timing:timing+donlen] += donsound
                    elif stamp[1] in (2, 4):
                        self.data[timing:timing +
                                  kalen] += kasound
                except ValueError:
                    pass
        
        elif diff == 'don':
            if isinstance(self.timestamp[0], tuple):
                for stamp in self.timestamp:
                    if stamp*self.samplerate+donlen < self.data.shape[0]:
                        self.data[int(stamp[0]*self.samplerate):int(stamp[0]*self.samplerate)+donlen] += donsound
            else:
                for stamp in self.timestamp:
                    if stamp*self.samplerate+donlen < self.data.shape[0]:
                        self.data[int(stamp*self.samplerate):int(stamp*self.samplerate)+donlen] += donsound
        
        elif diff == 'ka':
            if isinstance(self.timestamp[0], tuple):
                for stamp in self.timestamp:
                    if stamp*self.samplerate+kalen < self.data.shape[0]:
                        self.data[int(stamp[0]*self.samplerate):int(stamp[0]*self.samplerate)+kalen] += kasound
            else:
                for stamp in self.timestamp:
                    if stamp*self.samplerate+kalen < self.data.shape[0]:
                        self.data[int(stamp*self.samplerate):int(stamp*self.samplerate)+kalen] += kasound


def make_frame(data, nhop, nfft):
    """
    helping function for fftandmelscale.
    細かい時間に切り分けたものを学習データとするため，nhop(512)ずつずらしながらnfftサイズのデータを配列として返す
    """
    
    length = data.shape[0]
    framedata = np.concatenate((data, np.zeros(nfft)))  # zero padding
    return np.array([framedata[i*nhop:i*nhop+nfft] for i in range(length//nhop)])  


@jit
def fft_and_melscale(song, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):
    """
    fft and melscale method.
    fft: nfft = [1024, 2048, 4096]; サンプルの切り取る長さを変えながらデータからnp.arrayを抽出して高速フーリエ変換を行う．
    melscale: 周波数の次元を削減するとともに，log10の値を取っている．
    """

    feat_channels = []
    
    for nfft in nffts:
        
        feats = []
        window = signal.blackmanharris(nfft)
        filt = mel(song.samplerate, nfft, mel_nband, mel_freqlo, mel_freqhi)
        
        # get normal frame
        frame = make_frame(song.data, nhop, nfft)
        print(frame.shape)

        # melscaling
        processedframe = fft(window*frame)[:, :nfft//2+1]
        processedframe = np.dot(filt, np.transpose(np.abs(processedframe)**2))
        processedframe = 20*np.log10(processedframe+0.1)
        print(processedframe.shape)

        feat_channels.append(processedframe)
    
    if include_zero_cross:
        song.zero_crossing = np.where(np.diff(np.sign(song.data)))[0]
        print(song.zero_crossing)
    
    return np.array(feat_channels)


@jit(parallel=True)
def multi_fft_and_melscale(songs, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):
    
    for i in prange(len(songs)):
        songs[i].feats = fft_and_melscale(songs[i], nhop, nffts, mel_nband, mel_freqlo, mel_freqhi)


def music_for_listening(serv, synthesize=True, difficulty=0):

    song = Audio(glob(serv+"/*.ogg")[0])
    if synthesize:
        song.import_tja(glob(serv+"/*.tja")[-1], difficulty=difficulty)
        song.synthesize()
    plt.plot(song.data[1000:1512, 0])
    plt.show()
    song.save("./data/savedmusic.wav")


def music_for_validation(serv, deletemusic=True, verbose=False, difficulty=1):

    song = Audio(glob(serv+"/*.ogg")[0], stereo=False)
    song.import_tja(glob(serv+"/*.tja")[-1], difficulty=difficulty)
    song.feats = fft_and_melscale(song, nhop, nffts, mel_nband,
                        mel_freqlo, mel_freqhi, include_zero_cross=include_zero_cross)

    if deletemusic:
        song.data = None
    with open('./data/pickles/valdata.pickle', mode='wb') as f:
        pickle.dump(song, f)


def music_for_learning(serv, deletemusic=True, verbose=False, difficulty=0, diff=False, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):
    
    songplaces = glob(serv)
    songs = []
    for songplace in songplaces:
        if verbose:
            print(songplace)
        song = Audio(glob(songplace+"/*.ogg")[0])
        song.import_tja(glob(songplace+"/*.tja")
                       [-1], difficulty=difficulty, diff=True)
        song.data = (song.data[:, 0]+song.data[:, 1])/2
        songs.append(song)
    multi_fft_and_melscale(songs, nhop, nffts, mel_nband,
                        mel_freqlo, mel_freqhi, include_zero_cross=include_zero_cross)
    if deletemusic:
        for song in songs:
            song.data = None
    with open('./data/pickles/learningdata.pickle', mode='wb') as f:
        pickle.dump(songs, f)


def music_for_test(serv, deletemusic=True, verbose=False):

    song = Audio(glob(serv+"/*.ogg")[0], stereo=False)
    # song.import_tja(glob(serv+"/*.tja")[-1])
    song.feats = fft_and_melscale(song, include_zero_cross=False)
    with open('./data/pickles/testdata.pickle', mode='wb') as f:
        pickle.dump(song, f)


if __name__ == "__main__":
    
    """
    Test:
        >>>serv = "./data/songs/"
        >>>music_for_test(serv)
    Validation:
        >>>music_for_validation(serv, difficulty=0)
    Listening:
        >>>music_for_listening(serv)
    Learning:
        >>>serv = "./taitatsudata/*"
        >>>music_for_learning(serv, verbose=True, difficulty=0, diff=True)
    """

    serv = "./data/songs"
    music_for_test(serv)
