import pickle
import numpy as np
from scipy.signal import argrelmax
from librosa.util import peak_pick
from librosa.onset import onset_detect
from music_processor import *

def detection(don_inference, ka_inference, song):
    """detects notes disnotesiresultg don and ka"""

    don_inference = smooth(don_inference, 5)
    ka_inference = smooth(ka_inference, 5)

    don_timestamp = (peak_pick(don_inference, 1, 2, 4, 5, 0.05, 3)+7)  # 実際は7フレーム目のところの音
    ka_timestamp = (peak_pick(ka_inference, 1, 2, 4, 5, 0.05, 3)+7)
    
    print(don_timestamp)
    print(ka_timestamp)

    song.don_timestamp = don_timestamp[np.where(don_inference[don_timestamp] > ka_inference[don_timestamp])]
    song.timestamp = song.don_timestamp*512/song.samplerate
    # print(len(song.timestamp))
    song.synthesize(diff='don')

    # song.ka_timestamp = song.don_timestamp
    song.ka_timestamp = ka_timestamp[np.where(ka_inference[ka_timestamp] > don_inference[ka_timestamp])]
    song.timestamp=song.ka_timestamp*512/song.samplerate
    # print(len(song.timestamp))
    song.synthesize(diff='ka')

    song.save("./data/predict/created_music.wav")


def create_tja(filename, song, don_timestamp, ka_timestamp=None):

    if ka_timestamp is None:
        timestamp=don_timestamp*512/song.samplerate
        with open(filename, "w") as f:
            f.write('TITLE: xxx\nSUBTITLE: --\nBPM: 240\nWAVE:xxx.ogg\nOFFSET:0\n#START\n')
            i = 0
            time = 0
            while(i < len(timestamp)):
                if time/100 >= timestamp[i]:
                    f.write('1')
                    i += 1
                else:
                    f.write('0')
                if time % 100 == 99:
                    f.write(',\n')
                time += 1
            f.write('#END')

    else:
        don_timestamp=np.rint(don_timestamp*512/song.samplerate*100).astype(np.int32)
        ka_timestamp=np.rint(ka_timestamp*512/song.samplerate*100).astype(np.int32)
        with open(filename, "w") as f:
            f.write('TITLE: xxx\nSUBTITLE: --\nBPM: 240\nWAVE:xxx.ogg\nOFFSET:0\n#START\n')
            for time in range(np.max((don_timestamp[-1],ka_timestamp[-1]))):
                if np.isin(time,don_timestamp) == True:
                    f.write('1')
                elif np.isin(time,ka_timestamp) == True:
                    f.write('2')
                else:
                    f.write('0')
                if time%100==99:
                    f.write(',\n')
            f.write('#END')


if __name__ == "__main__":
    
    with open('./data/pickles/test_data.pickle', mode='rb') as f:
        song = pickle.load(f)

    with open('./data/pickles/don_inference.pickle', mode='rb') as f:
        don_inference = pickle.load(f)

    with open('./data/pickles/ka_inference.pickle', mode='rb') as f:
        ka_inference = pickle.load(f)

    detection(don_inference, ka_inference, song)
    create_tja("./data/predict/result.tja",song, song.don_timestamp, song.ka_timestamp)

