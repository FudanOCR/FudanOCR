'''
字符表模块
提供两种字符表读入方式：
1.给定txt文本链接，从txt文本中读取，也可以提供一个链接数组
2.给定一个字符串，从字符串读取
'''

class Alphabet(object):

    def __init__(self, wordAddress=None, words=None):

        self.wordAddress = wordAddress
        self.str = ""

        if wordAddress != None:
            self.readTextFromAddress(wordAddress)
        if words != None:
            self.readTextFromWords(words)

    def __len__(self):
        return len(self.str)

    def getStr(self):
        return self.str

    def readTextFromWords(self,words):
        for char in words:
            if char not in self.str:
                self.str += char

    def readTextFromAddress(self, address):

        if isinstance(address, list):
            for add in self.wordAddress:
                self.readTextFromAddress(add)
        elif isinstance(address, str):
            f = open(address, 'r', encoding='utf-8')
            for line in f.readlines():
                line = line.strip()
                for char in line:
                    if char in self.str:
                        continue
                    else:
                        self.str += char
