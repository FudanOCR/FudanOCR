import torch.nn as nn
import time
import torch

class CompositeLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, multi_gpu=False):
        super(CompositeLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
        self.multi_gpu = multi_gpu
        initrange = 0.08
        print("Initializing Bidirectional LSTM...")
        for weight in self.rnn.parameters():
            weight.data.uniform_(-initrange, initrange)
    '''     
    def forward(self, input):
        #if self.multi_gpu:
            #self.rnn.flatten_parameters()
        residual=input
        start = time.time()
        recurrent, _ = self.rnn(input)
        #print('Recurrent Net cost: {:.3f}'.format(time.time() - start))

        T, b, h = recurrent.size()
        t_rec = recurrent.view(T*b, h)

        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        output = torch.cat((output,residual),dim=2)
        return output
    '''
    
    def forward(self, input):
        #if self.multi_gpu:
            #self.rnn.flatten_parameters()

        start = time.time()
        recurrent, _ = self.rnn(input)
        #print('Recurrent Net cost: {:.3f}'.format(time.time() - start))

        T, b, h = recurrent.size()
        t_rec = recurrent.view(T*b, h)

        output = self.embedding(t_rec)
        output = output.view(T, b, -1)

        return output
    
class MLayerLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nLayer, nClass, dropout, multi_gpu=False):
        super(MLayerLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, nLayer, dropout=dropout, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nClass)
        self.multi_gpu = multi_gpu
        initrange = 0.08
        print("Initializing Bidirectional LSTM...")
        for weight in self.rnn.parameters():
            weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        if self.multi_gpu:
            self.rnn.flatten_parameters()

        recurrent, _ = self.rnn(input)

        T, b, h = recurrent.size()
        t_rec = recurrent.view(T*b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        
        return output
'''    
def compositelstm(rnn_conf, n_class):
    in_dim = rnn_conf['n_In']
    n_hidden = rnn_conf['n_Hidden']
    multi_gpu = rnn_conf['multi_gpu']
    model = nn.Sequential(
        CompositeLSTM(in_dim, n_hidden, 2*n_hidden, multi_gpu),
        CompositeLSTM(in_dim+2*n_hidden, n_hidden, 2*n_hidden, multi_gpu),
        nn.Linear(in_dim + 4*n_hidden, n_class)
    )
    return model

'''
def compositelstm(rnn_conf, n_class):
    in_dim = rnn_conf['n_In']
    n_hidden = rnn_conf['n_Hidden']
    multi_gpu = rnn_conf['multi_gpu']
    model = nn.Sequential(
        CompositeLSTM(in_dim, n_hidden, n_hidden, multi_gpu),
        CompositeLSTM(n_hidden, n_hidden, n_class, multi_gpu)
    )
    return model

def lstm_2layer(rnn_conf, n_class):
    in_dim = rnn_conf['n_In']
    n_hidden = rnn_conf['n_Hidden']
    n_layer = rnn_conf['n_Layer']
    dropout = rnn_conf['dropout']
    multi_gpu = rnn_conf['multi_gpu']
    model = MLayerLSTM(in_dim, n_hidden, n_layer, n_class, dropout, multi_gpu)
    return model


#TODO Implement Seq2Seq model
#class Seq2Seq(nn.Module):
    



