import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torchvision



'''
Decoupled Text Decoder
'''
class DTD(nn.Module):
    # LSTM DTD
    def __init__(self, nclass, nchannel, dropout = 0.3):
        super(DTD,self).__init__()
        self.nclass = nclass
        self.nchannel = nchannel
        self.pre_lstm = nn.LSTM(nchannel, int(nchannel / 2), bidirectional=True)
        self.rnn = nn.GRUCell(nchannel * 2, nchannel)
        self.generator = nn.Sequential(
                            nn.Dropout(p = dropout),
                            nn.Linear(nchannel, nclass)
                        )
        self.char_embeddings = Parameter(torch.randn(nclass+1, nchannel))

    def forward(self, feature, A, text, text_length, test = False):
        nB, nC, nH, nW = feature.size()
        nT = A.size()[1]
        # Normalize
        A = A / A.view(nB, nT, -1).sum(2).view(nB,nT,1,1)
        # weighted sum
        C = feature.view(nB, 1, nC, nH, nW) * A.view(nB, nT, 1, nH, nW)
        C = C.view(nB,nT,nC,-1).sum(3).transpose(1,0)
        C, _ = self.pre_lstm(C)
        C = F.dropout(C, p = 0.3, training=self.training)
        if not test:

            lenText = int(text_length.sum())
            nsteps = int(text_length.max())

            targets = torch.zeros(nB, nsteps + 1).long()
            if self.cuda:
                targets = targets.cuda()
            start_id = 0

            for i in range(nB):
                targets[i][1:1 + text_length.data[i]] = text.data[start_id:start_id + text_length.data[i]]+1
                start_id = start_id + text_length.data[i]
            targets = Variable(targets.transpose(0, 1).contiguous())





            gru_res = torch.zeros(C.size()).type_as(C.data)
            out_res = torch.zeros(lenText, self.nclass).type_as(feature.data)
            out_attns = torch.zeros(lenText, nH, nW).type_as(A.data)

            hidden = torch.zeros(nB, self.nchannel).type_as(C.data)
            prev_emb = self.char_embeddings.index_select(0, torch.zeros(nB).long().type_as(text.data))
            for i in range(0, nsteps):
                # print(C.size(),prev_emb.size(),nB)

                hidden = self.rnn(torch.cat((C[i, :, :], prev_emb), dim = 1),
                                 hidden)
                gru_res[i, :, :] = hidden
                prev_emb = self.char_embeddings.index_select(0, targets[i])
            gru_res = self.generator(gru_res)

            start = 0
            for i in range(0, nB):
                cur_length = int(text_length[i])
                out_res[start : start + cur_length] = gru_res[0: cur_length,i,:]
                out_attns[start : start + cur_length] = A[i,0:cur_length,:,:]
                start += cur_length

            return out_res


        else:
            lenText = nT
            nsteps = nT
            out_res = torch.zeros(lenText, nB, self.nclass).type_as(feature.data)

            hidden = torch.zeros(nB, self.nchannel).type_as(C.data)
            prev_emb = self.char_embeddings.index_select(0, torch.zeros(nB).long().type_as(text.data))
            out_length = torch.zeros(nB)
            now_step = 0
            while 0 in out_length and now_step < nsteps:
                hidden = self.rnn(torch.cat((C[now_step, :, :], prev_emb), dim = 1),
                                 hidden)
                tmp_result = self.generator(hidden)
                out_res[now_step] = tmp_result
                tmp_result = tmp_result.topk(1)[1].squeeze()
                for j in range(nB):
                    if out_length[j] == 0 and tmp_result[j] == 0:
                        out_length[j] = now_step + 1
                prev_emb = self.char_embeddings.index_select(0, tmp_result)
                now_step += 1
            for j in range(0, nB):
                if int(out_length[j]) == 0:
                    out_length[j] = nsteps

            start = 0
            output = torch.zeros(int(out_length.sum()), self.nclass).type_as(feature.data)
            for i in range(0, nB):
                cur_length = int(out_length[i])
                output[start : start + cur_length] = out_res[0: cur_length,i,:]
                start += cur_length

            return output