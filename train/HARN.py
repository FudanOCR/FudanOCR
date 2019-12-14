# -- coding: utf-8 --
def train_HARN(config_file):

    import sys
    sys.path.append('./recognition_model/HARN')

    import argparse
    import os
    import random
    import io
    import sys
    import time
    from models.moran import MORAN
    import tools.utils as utils
    import torch.optim as optim
    import numpy as np
    import torch.backends.cudnn as cudnn
    import torch.utils.data
    import tools.dataset as dataset
    from torch.autograd import Variable
    from collections import OrderedDict
    from tools.logger import logger
    from wordlist import result
    # from wordlistlsvt import result

    import warnings
    warnings.filterwarnings('ignore')

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = '5'  # 指定GPU

    from yacs.config import CfgNode as CN

    def read_config_file(config_file):
        # 用yaml重构配置文件
        f = open(config_file)
        opt = CN.load_cfg(f)
        return opt

    opt = read_config_file(config_file)  # 获取了yaml文件

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # Modify
    opt.alphabet = result

    assert opt.ngpu == 1, "Multi-GPU training is not supported yet, due to the variant lengths of the text in a batch."

    if opt.experiment is None:
        opt.experiment = 'expr'
    os.system('mkdir {0}'.format(opt.experiment))

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    # ---------save logger---------#
    log = logger('/home/msy/HARN_Moran/logger/asrn_se50_OCRdata_50')
    # log = logger('./logger/asrn_se50_lsvt_50')     # # 保存日志的路径 / 需要改
    # -----------------------------#

    if not torch.cuda.is_available():
        assert not opt.cuda, 'You don\'t have a CUDA device.'

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    train_nips_dataset = dataset.lmdbDataset(root=opt.train_nips,
                                             transform=dataset.resizeNormalize((opt.imgW, opt.imgH)),
                                             reverse=opt.BidirDecoder)
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
                                       transform=dataset.resizeNormalize((opt.imgW, opt.imgH)),
                                       reverse=opt.BidirDecoder)
    nclass = len(opt.alphabet.split(opt.sep))  # 一共有多少类，英文是36，中文就是wordlist，系统只认名字为wordlist.py的文件，记得将需要用的文件改为这个名字
    nc = 1

    converter = utils.strLabelConverterForAttention(opt.alphabet,
                                                    opt.sep)  # 给每个字一个编号，例如：中(2)国(30)人(65)；convert是id和字符之间的转换
    criterion = torch.nn.CrossEntropyLoss()

    if opt.cuda:
        MORAN = MORAN(nc, nclass, opt.nh, opt.targetH, opt.targetW, BidirDecoder=opt.BidirDecoder, CUDA=opt.cuda,
                      log=log)
    else:
        MORAN = MORAN(nc, nclass, opt.nh, opt.targetH, opt.targetW, BidirDecoder=opt.BidirDecoder,
                      inputDataType='torch.FloatTensor', CUDA=opt.cuda, log=log)

    if opt.MORAN != '':
        print('loading pretrained model from %s' % opt.MORAN)
        if opt.cuda:
            state_dict = torch.load(opt.MORAN)
        else:
            state_dict = torch.load(opt.MORAN, map_location='cpu')
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove `module.`
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

    image = Variable(image)  # 把图片转换成 CUDA 可以识别的 Variable 变量
    text = Variable(text)
    text_rev = Variable(text_rev)
    length = Variable(length)

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer  # 优化器的选择，这里用的Adam
    if opt.adam:
        optimizer = optim.Adam(MORAN.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.adadelta:
        optimizer = optim.Adadelta(MORAN.parameters(), lr=opt.lr)
    elif opt.sgd:
        optimizer = optim.SGD(MORAN.parameters(), lr=opt.lr, momentum=0.9)
    else:
        optimizer = optim.RMSprop(MORAN.parameters(), lr=opt.lr)

    def levenshtein(s1, s2):  # 莱温斯坦距离，编辑距离的一种
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


    def val(dataset, criterion, max_iter=10000, steps=None):
        data_loader = torch.utils.data.DataLoader(
            dataset, shuffle=False, batch_size=opt.batchSize, num_workers=int(opt.workers))  # opt.batchSize
        val_iter = iter(data_loader)
        max_iter = min(max_iter, len(data_loader))
        n_correct = 0
        n_total = 0
        distance = 0.0
        loss_avg = utils.averager()

        # f = open('./log.txt', 'a', encoding='utf-8')

        for i in range(max_iter):  # 设置很大的循环数值（达不到此值就会收敛）
            data = val_iter.next()
            if opt.BidirDecoder:
                cpu_images, cpu_texts, cpu_texts_rev = data  # data是dataloader导入的东西
                utils.loadData(image, cpu_images)
                t, l = converter.encode(cpu_texts, scanned=True)  # 这个encode是将字符encode成id
                t_rev, _ = converter.encode(cpu_texts_rev, scanned=True)
                utils.loadData(text, t)
                utils.loadData(text_rev, t_rev)
                utils.loadData(length, l)
                preds0, preds1 = MORAN(image, length, text, text_rev, debug=False, test=True, steps=steps)  # 跑模型HARN
                cost = criterion(torch.cat([preds0, preds1], 0), torch.cat([text, text_rev], 0))
                preds0_prob, preds0 = preds0.max(1)  # 取概率最大top1的结果
                preds0 = preds0.view(-1)
                preds0_prob = preds0_prob.view(-1)  # 维度的变形（好像是
                sim_preds0 = converter.decode(preds0.data, length.data)  # 将 id decode为字
                preds1_prob, preds1 = preds1.max(1)
                preds1 = preds1.view(-1)
                preds1_prob = preds1_prob.view(-1)
                sim_preds1 = converter.decode(preds1.data, length.data)
                sim_preds = []  # 预测出来的字
                for j in range(cpu_images.size(0)):  # 对字典进行处理，把单个字符连成字符串
                    text_begin = 0 if j == 0 else length.data[:j].sum()
                    if torch.mean(preds0_prob[text_begin:text_begin + len(sim_preds0[j].split('$')[0] + '$')]).item() > \
                            torch.mean(
                                preds1_prob[text_begin:text_begin + len(sim_preds1[j].split('$')[0] + '$')]).item():
                        sim_preds.append(sim_preds0[j].split('$')[0] + '$')
                    else:
                        sim_preds.append(sim_preds1[j].split('$')[0][-1::-1] + '$')
            else:  # 用不到的另一种情况
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

            loss_avg.add(cost)  # 计算loss的平均值
            for pred, target in zip(sim_preds, cpu_texts):  # 与GroundTruth的对比，cpu_texts是GroundTruth，sim_preds是连接起来的字符串
                if pred == target.lower():  # 完全匹配量
                    n_correct += 1
                # f.write("pred %s\t      target %s\n" % (pred, target))
                distance += levenshtein(pred, target) / max(len(pred), len(target))  # 莱温斯坦距离
                n_total += 1  # 完成了一个单词

        # f.close()

        # print and save     # 跑完之后输出到日志中
        for pred, gt in zip(sim_preds, cpu_texts):
            gt = ''.join(gt.split(opt.sep))
            print('%-20s, gt: %-20s' % (pred, gt))

        print("correct / total: %d / %d, " % (n_correct, n_total))
        print('levenshtein distance: %f' % (distance / n_total))
        accuracy = n_correct / float(n_total)
        log.scalar_summary('Validation/levenshtein distance', distance / n_total, steps)
        log.scalar_summary('Validation/loss', loss_avg.val(), steps)
        log.scalar_summary('Validation/accuracy', accuracy, steps)
        print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
        return accuracy


    def trainBatch(steps):
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
        cost.backward()  # 反向传播
        optimizer.step()  # 优化器
        return cost

    t0 = time.time()
    acc, acc_tmp = 0, 0
    print(' === HARN === ')

    for epoch in range(opt.niter):
        print(" === Loading Train Data === ")
        train_iter = iter(train_loader)
        i = 0
        print(" === start training === ")
        while i < len(train_loader):  # len()：数据大小
            # print("main函数里,可迭代次数为 %d" %  len(train_loader))
            steps = i + epoch * len(train_loader)  # step用来计算什么时候进行存储/打印
            if steps % opt.valInterval == 0:
                for p in MORAN.parameters():
                    p.requires_grad = False
                MORAN.eval()
                print('---------------Please Waiting----------------')  # train的一些打印信息
                acc_tmp = val(test_dataset, criterion, steps=steps)
                if acc_tmp > acc:
                    acc = acc_tmp
                    try:
                        time.sleep(0.01)
                        torch.save(MORAN.state_dict(), '{0}/{1}_{2}.pth'.format(opt.experiment, i, str(acc)[:6]))
                        print(".pth")
                    except RuntimeError:
                        print("RuntimeError")
                        pass

            for p in MORAN.parameters():
                p.requires_grad = True
            MORAN.train()

            cost = trainBatch(steps)
            loss_avg.add(cost)


            t1 = time.time()  # niter是参数部分设置的epoch数量
            print('Epoch: %d/%d; iter: %d/%d; Loss: %f; time: %.2f s;' %
                  (epoch, opt.niter, i, len(train_loader), loss_avg.val(), t1 - t0)),
            log.scalar_summary('train loss', loss_avg.val(), steps)  # 拟合到90多/拟合到1，完全收敛，训练充分
            log.scalar_summary('speed batches/persec', steps / (time.time() - t0), steps)
            loss_avg.reset()
            t0 = time.time()

            '''
            if i % 100 == 0:
                t1 = time.time()  # niter是参数部分设置的epoch数量
                print('Epoch: %d/%d; iter: %d/%d; Loss: %f; time: %.2f s;' %
                      (epoch, opt.niter, i, len(train_loader), loss_avg.val(), t1 - t0)),
                log.scalar_summary('train loss', loss_avg.val(), i)  # 拟合到90多/拟合到1，完全收敛，训练充分
                log.scalar_summary('speed batches/persec', i / (time.time() - t0), i)
                loss_avg.reset()
                t0 = time.time()
            '''
            '''
            if steps % opt.displayInterval == 0:
                t1 = time.time()     # niter是参数部分设置的epoch数量
                print('Epoch: %d/%d; iter: %d/%d; Loss: %f; time: %.2f s;' %
                      (epoch, opt.niter, i, len(train_loader), loss_avg.val(), t1 - t0)),
                log.scalar_summary('train loss', loss_avg.val(), steps)     # 拟合到90多/拟合到1，完全收敛，训练充分
                log.scalar_summary('speed batches/persec', steps / (time.time() - t0), steps)
                loss_avg.reset()
                t0 = time.time()
            '''

            i += 1



