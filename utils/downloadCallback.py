def callbackfunc(blocknum, blocksize, totalsize):
    '''回调函数
    @blocknum: 已经下载的数据块
    @blocksize: 数据块的大小
    @totalsize: 远程文件的大小
    '''
    percent = 100.0 * blocknum * blocksize / totalsize

    nowLoad = blocknum * blocksize / 1024 / 1024
    total = totalsize / 1024 / 1024

    if percent > 100:
        percent = 100
    print("\r %.2fM/%.2fM    %.2f%%" % (nowLoad, total, percent), end=" ")