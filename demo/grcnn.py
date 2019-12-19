# -*- coding: utf-8 -*-

def demo_grcnn(config_yaml):

    import sys
    sys.path.append('./recognition_model/GRCNN')

    import torch
    import os
    from utils import keys
    from models import crann
    import dataset
    from utils import util
    import torch.nn.functional as F
    import io
    import yaml
    import tools.utils as utils
    import tools.dataset_lmdb as dataset_lmdb
    import torchvision.transforms as transforms
    import lmdb

    # 需要在配置文件里体现
    # opt.model_path = 'checkpoints/grcnn_art/crann_11_1.pth'
    # batch_size = 16
    #imgH = 32
    # maxW = 100
    # num_workers = 4
    # cnn_model = 'grcnn'
    # rnn_model = 'compositelstm'
    # n_In = 512
    # n_Hidden = 256
    # test_set = '../art_test.txt'

    # from yacs.config import CfgNode as CN
    #
    # def read_config_file(config_file):
    #     # 用yaml重构配置文件
    #     f = open(config_file)
    #     opt = CN.load_cfg(f)
    #     return opt
    #
    # opt = read_config_file(config_file)

    f = open(config_yaml, encoding='utf-8')
    opt = yaml.load(f)



    alphabet = keys.alphabet
    nClass = len(alphabet) + 1
    converter = util.strLabelConverter(alphabet)

    model = crann.CRANN(opt, nClass).cuda()
    if os.path.isfile(opt['DEMO']['model_path']):
        print("=> loading checkpoint '{}'".format(opt['DEMO']['model_path']))
        checkpoint = torch.load(opt['DEMO']['model_path'])
        start_epoch = checkpoint['epoch']
        # best_pred = checkpoint['best_pred']
        model.load_state_dict(checkpoint['state_dict'])
        # print("=> loaded checkpoint '{}' (epoch {} accuracy {})"
        #       .format(opt.model_path, checkpoint['epoch'], best_pred))

    model.eval()

    # root, mappinggit

    train_set = dataset_lmdb.lmdbDataset(opt['DEMO']['test_set_lmdb'],transform=dataset_lmdb.resizeNormalize((opt['TRAIN']['MAX_W'], opt['TRAIN']['IMG_H'])))

    # train_set = dataset.testDataset(opt['test_set'])  # dataset.graybackNormalize()
    test_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size=opt['TRAIN']['BATCH_SIZE'],
                                              shuffle=False,
                                              num_workers=opt['WORKERS'],)
                                              # collate_fn=dataset.alignCollate(
                                              #     imgH=opt['TRAIN']['IMG_H'],
                                              #     imgW=opt['TRAIN']['MAX_W']))

    file = open('./pred.txt', 'w', encoding='utf-8')
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