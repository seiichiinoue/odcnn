from model import *
import pickle

if __name__ == '__main__':
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = convNet()
    net = net.to(device)

    with open('./data/pickles/test_data.pickle', mode='rb') as f:
        song = pickle.load(f)


    if sys.argv[1] == 'don':
        
        if torch.cuda.is_available():
            net.load_state_dict(torch.load('./models/don_model.pth'))
        else:
            net.load_state_dict(torch.load('./models/don_model.pth', map_location='cpu'))

        inference = net.infer(song.feats, device, minibatch=4192)
        inference = np.reshape(inference, (-1))

        with open('./data/pickles/don_inference.pickle', mode='wb') as f:
            pickle.dump(inference, f)

    if sys.argv[1] == 'ka':
        
        if torch.cuda.is_available():
            net.load_state_dict(torch.load('./models/ka_model.pth'))
        else:
            net.load_state_dict(torch.load('./models/ka_model.pth', map_location='cpu'))

        inference = net.infer(song.feats, device, minibatch=4192)
        inference = np.reshape(inference, (-1))

        with open('./data/pickles/ka_inference.pickle', mode='wb') as f:
            pickle.dump(inference, f)
