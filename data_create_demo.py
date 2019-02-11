from music_processor import *
from glob import glob
import pickle

print("target file: ", glob('data/songs'+'/*.ogg')[0])
song = Audio(glob('data/songs'+'/*.ogg')[0])
song.import_tja(glob('data/songs'+'/*.tja')[0])

print("song data: shape: ",song.data.shape)

song.data = (song.data[:,0]+song.data[:,1])/2
song.feats = fft_and_melscale(song, include_zero_cross=False)

print("feats: shape: ", song.feats.shape)

# save file as pickle format
with open('./data/pickles/testdata.pickle', mode="wb") as f:
    pickle.dump(song, f)

print("data import and save complete!")

# recover data from pickle
with open('./data/pickles/testdata.pickle', mode="rb") as f:
    song = pickle.load(f)

    print("loaded data from pickle successfully!")

