class Generator(object):
    """An abstract class for text generator.
    一个抽象类，用于数据集的生成，等待被实现
    """

    def __getitem__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class CnumberGenerator(Generator):
    def __init__(self):
        self.cnum = cnumber()

    def __len__(self):
        return 128000

    def __getitem__(self, index):
        num = random.randint(100, 9999999)
        if random.randint(0, 1):
            num = num / 100.0
        return self.cnum.cwchange(num)


class TextGenerator(Generator):
    """Invoice message txt generator
    args:
        texts: File path which contains
    """

    def __init__(self, texts, len_thr):
        super(TextGenerator, self).__init__()
        self.len_thr = len_thr
        with open(texts) as f:
            self.texts = f.readlines()

    def __getitem__(self, index):
        text_len = len(self.texts[index])
        if text_len > self.len_thr:
            text_len = self.len_thr
        return self.texts[index].strip()[0:text_len]

    def __len__(self):
        return len(self.texts)

    def __len_thr__(self):
        return self.len_thr


class PasswordGenerator(Generator):
    def __init__(self):
        self.fake = Faker()
        self.fake.random.seed(4323)

    def __getitem__(self, index):
        return self.fake.password(length=10, special_chars=True, digits=True, upper_case=True, lower_case=True)

    def __len__(self):
        return 320000


class HyperTextGenerator(Generator):
    def __init__(self, texts):
        self.invoice_gen = TextGenerator(texts)
        # self.passwd_gen = PasswordGenerator()
        self.cnum_gen = CnumberGenerator()

    def __getitem__(self, index):
        rnd = random.randint(0, 1)
        if rnd:
            cur = index % self.invoice_gen.__len__()
            return self.invoice_gen.__getitem__(cur)
        else:
            return self.cnum_gen.__getitem__(index)

    def __len__(self):
        return self.invoice_gen.__len__() + self.cnum_gen.__len__()
