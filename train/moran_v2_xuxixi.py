def train_moran_v2_xuxixi(config_file):

    import sys
    sys.path.append('./recognition_model/MORAN_V2_xuxixi')

    import argparse
    import random
    import torch
    import torch.backends.cudnn as cudnn
    import torch.optim as optim
    import torch.utils.data
    from torch.autograd import Variable
    import numpy as np
    import os
    import tools.utils as utils
    import tools.dataset as dataset
    import time
    from collections import OrderedDict
    from models.moran import MORAN
    from wordlist import result
    from yacs.config import CfgNode as CN

    def read_config_file(config_file):
        # 用yaml重构配置文件
        f = open(config_file)
        opt = CN.load_cfg(f)
        return opt

    opt = read_config_file(config_file)

    assert opt.ngpu == 1, "Multi-GPU training is not supported yet, due to the variant lengths of the text in a batch."

    if opt.experiment is None:
        opt.experiment = 'expr'
    os.system('mkdir {0}'.format(opt.experiment))

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if not torch.cuda.is_available():
        assert not opt.cuda, 'You don\'t have a CUDA device.'

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    train_nips_dataset = dataset.lmdbDataset(root=opt.train_nips,
        transform=dataset.resizeNormalize((opt.imgW, opt.imgH)), reverse=opt.BidirDecoder)
    assert train_nips_dataset
    '''
    train_cvpr_dataset = dataset.lmdbDataset(root=opt.train_cvpr,
        transform=dataset.resizeNormalize((opt.imgW, opt.imgH)), reverse=opt.BidirDecoder)
    assert train_cvpr_dataset
    '''
    '''
    train_dataset = torch.utils.data.ConcatDataset([train_nips_dataset, train_cvpr_dataset])
    '''
    train_dataset = train_nips_dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batchSize,
        shuffle=False, sampler=dataset.randomSequentialSampler(train_dataset, opt.batchSize),
        num_workers=int(opt.workers))

    test_dataset = dataset.lmdbDataset(root=opt.valroot,
        transform=dataset.resizeNormalize((opt.imgW, opt.imgH)), reverse=opt.BidirDecoder)

    nclass = len(opt.alphabet.split(opt.sep))
    nc = 1

    converter = utils.strLabelConverterForAttention(opt.alphabet, opt.sep)
    criterion = torch.nn.CrossEntropyLoss()

    if opt.cuda:
        MORAN = MORAN(nc, nclass, opt.nh, opt.targetH, opt.targetW, BidirDecoder=opt.BidirDecoder, CUDA=opt.cuda)
    else:
        MORAN = MORAN(nc, nclass, opt.nh, opt.targetH, opt.targetW, BidirDecoder=opt.BidirDecoder, inputDataType='torch.FloatTensor', CUDA=opt.cuda)

    if opt.MORAN != '':
        print('loading pretrained model from %s' % opt.MORAN)
        if opt.cuda:
            state_dict = torch.load(opt.MORAN)
        else:
            state_dict = torch.load(opt.MORAN, map_location='cpu')
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "") # remove `module.`
            MORAN_state_dict_rename[name] = v
        MORAN.load_state_dict(MORAN_state_dict_rename, strict=True)

    image = torch.FloatTensor(opt.batchSize, nc, opt.imgH, opt.imgW)
    text = torch.LongTensor(opt.batchSize * 5)
    text_rev = torch.LongTensor(opt.batchSize * 5)
    length = torch.IntTensor(opt.batchSize)

    if opt.cuda:
        MORAN.cuda()
        MORAN = torch.nn.DataParallel(MORAN, device_ids=range(opt.ngpu))
        image = image.cuda()
        text = text.cuda()
        text_rev = text_rev.cuda()
        criterion = criterion.cuda()

    image = Variable(image)
    text = Variable(text)
    text_rev = Variable(text_rev)
    length = Variable(length)

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(MORAN.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.adadelta:
        optimizer = optim.Adadelta(MORAN.parameters(), lr=opt.lr)
    elif opt.sgd:
        optimizer = optim.SGD(MORAN.parameters(), lr=opt.lr, momentum=0.9)
    else:
        optimizer = optim.RMSprop(MORAN.parameters(), lr=opt.lr)


    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)

        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]





    def val(dataset, criterion, max_iter=1000):
        print('Start val')
        data_loader = torch.utils.data.DataLoader(
            dataset, shuffle=False, batch_size=opt.batchSize, num_workers=int(opt.workers)) # opt.batchSize
        val_iter = iter(data_loader)
        max_iter = min(max_iter, len(data_loader))
        n_correct = 0
        n_total = 0
        distance = 0.0
        loss_avg = utils.averager()

        f = open('./log.txt','a',encoding='utf-8')

        for i in range(max_iter):
            data = val_iter.next()
            if opt.BidirDecoder:
                cpu_images, cpu_texts, cpu_texts_rev = data
                utils.loadData(image, cpu_images)
                t, l = converter.encode(cpu_texts, scanned=True)
                t_rev, _ = converter.encode(cpu_texts_rev, scanned=True)
                utils.loadData(text, t)
                utils.loadData(text_rev, t_rev)
                utils.loadData(length, l)
                preds0, preds1 = MORAN(image, length, text, text_rev, test=True)
                cost = criterion(torch.cat([preds0, preds1], 0), torch.cat([text, text_rev], 0))
                preds0_prob, preds0 = preds0.max(1)
                preds0 = preds0.view(-1)
                preds0_prob = preds0_prob.view(-1)
                sim_preds0 = converter.decode(preds0.data, length.data)
                preds1_prob, preds1 = preds1.max(1)
                preds1 = preds1.view(-1)
                preds1_prob = preds1_prob.view(-1)
                sim_preds1 = converter.decode(preds1.data, length.data)
                sim_preds = []
                for j in range(cpu_images.size(0)):
                    text_begin = 0 if j == 0 else length.data[:j].sum()
                    if torch.mean(preds0_prob[text_begin:text_begin+len(sim_preds0[j].split('$')[0]+'$')]).data[0] >\
                     torch.mean(preds1_prob[text_begin:text_begin+len(sim_preds1[j].split('$')[0]+'$')]).data[0]:
                        sim_preds.append(sim_preds0[j].split('$')[0]+'$')
                    else:
                        sim_preds.append(sim_preds1[j].split('$')[0][-1::-1]+'$')
            else:
                cpu_images, cpu_texts = data
                utils.loadData(image, cpu_images)
                t, l = converter.encode(cpu_texts, scanned=True)
                utils.loadData(text, t)
                utils.loadData(length, l)
                preds = MORAN(image, length, text, text_rev, test=True)
                cost = criterion(preds, text)
                _, preds = preds.max(1)
                preds = preds.view(-1)
                sim_preds = converter.decode(preds.data, length.data)

            loss_avg.add(cost)
            for pred, target in zip(sim_preds, cpu_texts):
                if pred == target.lower():
                    n_correct += 1
                f.write("预测 %s      目标 %s\n" % ( pred,target ) )
                distance += levenshtein(pred,target) / max(len(pred),len(target))
                n_total += 1

        f.close()

        print("correct / total: %d / %d, "  % (n_correct, n_total))
        print('levenshtein distance: %f' % (distance/n_total))

        accuracy = n_correct / float(n_total)
        print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
        return accuracy

    def trainBatch():
        data = train_iter.next()
        if opt.BidirDecoder:
            cpu_images, cpu_texts, cpu_texts_rev = data
            utils.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts, scanned=True)
            t_rev, _ = converter.encode(cpu_texts_rev, scanned=True)
            utils.loadData(text, t)
            utils.loadData(text_rev, t_rev)
            utils.loadData(length, l)
            preds0, preds1 = MORAN(image, length, text, text_rev)
            cost = criterion(torch.cat([preds0, preds1], 0), torch.cat([text, text_rev], 0))
        else:
            cpu_images, cpu_texts = data
            utils.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts, scanned=True)
            utils.loadData(text, t)
            utils.loadData(length, l)
            preds = MORAN(image, length, text, text_rev)
            cost = criterion(preds, text)

        MORAN.zero_grad()
        cost.backward()
        optimizer.step()
        return cost

    t0 = time.time()
    acc = 0
    acc_tmp = 0
    for epoch in range(opt.niter):

        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            # print("main函数里,可迭代次数为 %d" %  len(train_loader))

            if i % opt.valInterval == 0:
                for p in MORAN.parameters():
                    p.requires_grad = False
                MORAN.eval()

                acc_tmp = val(test_dataset, criterion)
                if acc_tmp > acc:
                    acc = acc_tmp
                    torch.save(MORAN.state_dict(), '{0}/{1}_{2}.pth'.format(
                            opt.experiment, i, str(acc)[:6]))

            if i % opt.saveInterval == 0:
                torch.save(MORAN.state_dict(), '{0}/{1}_{2}.pth'.format(
                            opt.experiment, epoch, i))

            for p in MORAN.parameters():
                p.requires_grad = True
            MORAN.train()

            cost = trainBatch()
            loss_avg.add(cost)

            if i % opt.displayInterval == 0:
                t1 = time.time()
                print ('Epoch: %d/%d; iter: %d/%d; Loss: %f; time: %.2f s;' %
                        (epoch, opt.niter, i, len(train_loader), loss_avg.val(), t1-t0)),
                loss_avg.reset()
                t0 = time.time()

            i += 1