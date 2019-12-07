import torch
import os
from utils import keys
from models import crann
import dataset
from utils import util
import torch.nn.functional as F
import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

model_path = 'checkpoints/grcnn_art/crann_11_1.pth'
batch_size = 16
imgH = 32
maxW = 100
num_workers = 4
cnn_model = 'grcnn'
rnn_model = 'compositelstm'
n_In = 512
n_Hidden = 256
test_set = '../art_test.txt'

if __name__ == '__main__':
    alphabet = keys.alphabet
    nClass = len(alphabet) + 1
    converter = util.strLabelConverter(alphabet)

    model = crann.CRANN(cnn_model, rnn_model, n_In, n_Hidden, nClass).cuda()
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        # best_pred = checkpoint['best_pred']
        model.load_state_dict(checkpoint['state_dict'])
        # print("=> loaded checkpoint '{}' (epoch {} accuracy {})"
        #       .format(model_path, checkpoint['epoch'], best_pred))

    model.eval()

    train_set = dataset.imageDataset(test_set)  # dataset.graybackNormalize()
    test_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              collate_fn=dataset.alignCollate(
                                                  imgH=imgH,
                                                  imgW=maxW))

    file = open('logger/pred.txt', 'w', encoding='utf-8')
    index = 0
    for i, (cpu_images, _) in enumerate(test_loader):
        bsz = cpu_images.size(0)
        images = cpu_images.cuda()

        predict = model(images)
        predict_len = torch.IntTensor([predict.size(0)] * bsz)
        _, acc = predict.max(2)
        acc = acc.transpose(1, 0).contiguous().view(-1)
        prob, _ = F.softmax(predict, dim=2).max(2)
        probilities = torch.mean(prob, dim=1)
        sim_preds = converter.decode(acc.data, predict_len.data, raw=False)
        for probility, pred in zip(probilities, sim_preds):
            index += 1
            img_key = 'gt_%d' % index
            file.write('%s:\t\t\t\t%.3f%%\t%-20s\n' % (img_key, probility.item() * 100, pred))
    file.close()
