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
    elif model_name == 'RARE':
        from model.recognition_model.RARE.models import RARE
        return RARE
    elif model_name == 'PixelLink':
        from model.detection_model.PixelLink.net import Net
        return Net
    else:
        return None