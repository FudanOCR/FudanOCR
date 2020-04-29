def getModel(model_name):
    if model_name == 'MORAN':
        from model.recognition_model.MORAN_V2.models.moran import newMORAN
        return newMORAN
    elif model_name == 'GRCNN':
        from model.recognition_model.GRCNN.models.crann import newCRANN
        return newCRANN
    elif model_name == 'TEXTNET':
        from model.detection_model.TextSnake_pytorch.network.textnet import TextNet
        return TextNet
    elif model_name == 'AdvancedEAST':
        from model.detection_model.AdvancedEAST.network.AEast import East
        return East
    elif model_name == 'CRNN':
        from model.recognition_model.CRNN.models import CRNN
        return CRNN
    elif model_name == 'AON':
        from model.recognition_model.AON.models import AON
        return AON
    elif model_name == 'RARE':
        from model.recognition_model.RARE.models_cap_sr_pyramid import RARE
        return RARE
    elif model_name == 'SAR':
        from model.recognition_model.SAR.models import SAR
        return SAR
    elif model_name == 'CAPSOCR':
        from model.recognition_model.CAPSOCR.models import CAPSOCR
        return CAPSOCR
    elif model_name == 'CAPSOCR2':
        from model.recognition_model.CAPSOCR2.models import CAPSOCR2
        return CAPSOCR2
    elif model_name == 'DAN':
        from model.recognition_model.DAN.models import DAN
        return DAN
    elif model_name == 'PixelLink':
        from model.detection_model.PixelLink.net import Net
        return Net
    elif model_name == 'LSN':
        from model.detection_model.LSN.lib.model.networkFactory import ResNet50
        return ResNet50
    else:
        return None