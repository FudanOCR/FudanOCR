from data.transform import transforms as T

def build_transforms(cfg, is_train=True):

    transforms = None

    if cfg.BASE.MODEL == 'MORAN':
        transforms = T.resizerNormalsize((cfg.IMAGE.IMG_W, cfg.IMAGE.IMG_H))

    elif cfg.BASE.MODEL == 'GRCNN':
        transforms = T.resizerNormalsizeAndPadding(cfg.IMAGE.IMG_W, cfg.IMAGE.IMG_H)

    elif cfg.BASE.MODEL == "TextSnake":
        transform = NewAugmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds, maxlen=1280, minlen=512)

    elif cfg.BASE.MODEL == 'maskrcnn':
        transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    
    return transform
