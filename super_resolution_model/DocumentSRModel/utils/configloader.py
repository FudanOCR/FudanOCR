#-*- coding:utf-8 -*-


class ConfigLoader(object):
    def __init__(self, configpath, section):
        import configparser
        cfloader = configparser.ConfigParser()
        cfloader.readfp(open(configpath))

        for (name, value) in cfloader.items(section):
            setattr(self, name, value)
