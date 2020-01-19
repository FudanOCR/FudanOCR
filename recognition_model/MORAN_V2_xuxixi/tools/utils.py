'''
utils.py 定义了一系列的工具函数
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import collections

class strLabelConverterForAttention(object):
    """Convert between str and label.

    NOTE:
        Insert `EOS` to the alphabet for attention.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, sep):
        self._scanned_list = False
        self._out_of_list = ''
        self._ignore_case = True
        self.sep = sep
        self.alphabet = alphabet.split(sep)

        self.dict = {}
        for i, item in enumerate(self.alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[item] = i

    def scan(self, text):
        # print(text)
        text_tmp = text
        text = []
        for i in range(len(text_tmp)):
            text_result = ''
            for j in range(len(text_tmp[i])):
                chara = text_tmp[i][j].lower() if self._ignore_case else text_tmp[i][j]
                if chara not in self.alphabet:
                    # 在这里处理
                    if chara in self._out_of_list:
                        continue
                    else:
                        self._out_of_list += chara
                        file_out_of_list = open("out_of_list.txt", "a+")
                        file_out_of_list.write(chara + "\n")
                        file_out_of_list.close()
                        print('" %s " is not in alphabet...' % chara)
                        continue
                else:
                    text_result += chara
            text.append(text_result)
        text_result = tuple(text)
        self._scanned_list = True
        return text_result

    def encode(self, text, scanned=True):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        self._scanned_list = scanned
        if not self._scanned_list:
            text = self.scan(text)
            
        if isinstance(text, str):
            #*************************************

            text0 = []
            for char in text:
                if char == '\'':
                    pass
                try:
                    #print(type(char))
                    # if char !=
                    #print(char)
                    text0.append(self.dict[char.lower()])
                # 把所有找不到的字符都当成'0'
                except KeyError:
                    text0.append(self.dict['0'.lower()])
            text=text0

            '''
            for char in text:
                try:
                    print(char)
                    text = [
                        self.dict[char.lower() if self._ignore_case else char]
                    ]
                except KeyError:
                    print('KeyError')
            '''
            #text=list(text)
            #for ti in range(0,len(text)):
                #if text[ti] =

            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.LongTensor(text), torch.LongTensor(length))

    def decode(self, t, length):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            return ''.join([self.alphabet[i] for i in t])
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.LongTensor([l])))
                index += l
            return texts

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """
    '''
    一个用于对torch.Variable或者torch.Tensor求平均值的类
    '''

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def loadData(v, data):
    '''
    将数据data搬运到v中

    :param torch.Tensor v 目标张量
    :param torch.Tensor data 源张量
    '''
    v.data.resize_(data.size()).copy_(data)
