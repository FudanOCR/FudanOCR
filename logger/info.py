import os


def file_summary(path, file_name, content, encoding='utf8'):
    '''
    Args:
        path(string):the path to the file needed to write in.
        file_name(string):the name to the file. e.g.`test.txt`
        content(string):the string needed to write.
        encoding(string):the coding scheme of the file.
    '''
    if os.path.exists(path) is False:
        os.makedirs(path)
    f = open(os.path.join(path, file_name), 'a', encoding=encoding)
    f.write(content)
    f.close()
