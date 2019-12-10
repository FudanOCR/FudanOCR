import os
import torch


class BaseModel(object):
    def __init__(self, cfg):
        self.__current_device_mode = 'cpu'
        self.__mode = cfg['mode']
        self.__init_networks(cfg['networks'][self.__mode['networks']], cfg.get('state_dict_path'))
        self.change_device_mode(cfg['device_mode'])

    def __init_networks(self, network_dict, state_dict_path=None):
        self.__networks = {}
        
        for key, info in network_dict.items():
            print('===> Initialize the network "{}"'.format(key))
            network = self.init_network(key, info['type'], info['parameters'])
            self.__networks[key] = network
        
        if state_dict_path is not None and os.path.exists(state_dict_path):
            self.load_state_dict(torch.load(state_dict_path))

    def init_network(self, network_name, network_type, parameters):
        raise NotImplementedError()

    def __call__(self, **kwargs):
        raise NotImplementedError()

    def load_state_dict(self, model_state_dict):
        network_state_dict = model_state_dict['networks']
        for key, network in self.__networks.items():
            model_dict = network.state_dict()
            state_dict = network_state_dict[key]

            state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(state_dict)
            network.load_state_dict(model_dict)

    def state_dict(self):
        model_state_dict = {}
        network_state_dict = {}
        for key, network in self.__networks.items():
            network_state_dict[key] = network.state_dict()
        model_state_dict['networks'] = network_state_dict
        return model_state_dict

    @property
    def networks(self):
        return self.__networks

    @property
    def mode(self):
        return self.__mode
    
    @property
    def use_gpu(self):
        return True if self.__current_device_mode == 'gpu' else False

    def change_device_mode(self, mode='gpu'):
        if self.__current_device_mode == mode:
            return True

        if mode == 'gpu':
            if torch.cuda.is_available():
                for network in self.__networks.values():
                    network.cuda()
            else:
                return False
        elif mode == 'cpu':
            for network in self.__networks.values():
                network.cpu()
        
        self.__current_device_mode = mode
        return True

    def train(self):
        for network in self.networks.values():
            network.train()
        self.init_train_mode()

    def eval(self):
        for network in self.networks.values():
            network.eval()
        self.init_eval_mode()
    
    def init_train_mode(self):
        pass
    
    def init_eval_mode(self):
        pass


class BaseTester(object):
    def __init__(self, test_cfg, model_cfg):
        self.__load_basic_config(test_cfg['basic'])
        
        self.model = self.__init_model(model_cfg)
        self.data_loader = self.__init_data_loader(test_cfg['dataset'])

    def __load_basic_config(self, cfg):
        self.use_gpu = cfg.get('use_gpu', True) and torch.cuda.is_available()
        
        model_path = cfg.get('model_path')
        if model_path is None:
            model_dir = cfg.get('model_dir')
            model_name = cfg.get('model_name')
            model_path = os.path.join(model_dir, model_name)
        self.model_path = model_path

        self.result_dir = cfg.get('result_dir', 'results/test/default')

    def __init_model(self, cfg):
        model = self.init_model(cfg)
        if self.use_gpu:
            model.change_device_mode('gpu')
        model.eval()
        return model

    def __init_data_loader(self, cfg):
        return self.init_data_loader(cfg)

    def init_model(self, cfg):
        raise NotImplementedError()

    def init_data_loader(self, cfg):
        raise NotImplementedError()

    def run(self, **kwargs):
        raise NotImplementedError()

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            return None
        checkpoint_dict = torch.load(path)
        if checkpoint_dict.get('networks') is not None:
            model_dict = checkpoint_dict
        elif checkpoint_dict.get('model') is not None:
            model_dict = checkpoint_dict['model']
        self.model.load_state_dict(model_dict)
        return True