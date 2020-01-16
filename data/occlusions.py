

class  Occlusion(object):

    def __init__(self):
        raise NotImplementedError

    def __call__(self, img):
        raise NotImplementedError

class RainDrops(Occlusion):

    def __init__(self, number, positions , width, height) :
        self.number = number
        self.positions = positions


    def __call__(self, img):
        # convert [0,1] scale to pixels language
        # np.intp : Integer used for indexing
        positions = np.round(positions * img_size[0]).astype(np.intp)
        width     = np.round(width * img_size[0]).astype(np.intp)
        height    = np.round(height * img_size[1]).astype(np.intp)

        # create masks

        # x
        extended = [np.arange(h).tolist() for h in height]
        extended = np.array(functools_reduce_iconcat(extended))
        mask_x   = np.repeat(positions[:, 0].reshape(1, -1), height)
        mask_x  += extended
        mask_x   = np.repeat(mask_x, width)

        # y
        extended = [(np.repeat(np.arange(width), h)).reshape(width, -1).T.reshape(-1).tolist() for h in height]
        extended = np.array(functools_reduce_iconcat(extended))
        mask_y   = np.repeat(positions[:, 1].reshape(1, -1), height * width)
        mask_y  += extended

        # put the mask on the image
        img[mask_x % img_size[0], mask_y % img_size[1]] = 255

        # rgb to bgr  for opencv
        return img[:, :, ::-1]


class MovingBar(Occlusion):

    def __init__(self) :
        pass

    def __call__(self, img):
        pass

class RemovePixels(Occlusion):

    def __init__(self) :
        pass

    def __call__(self, img):
        pass

class Clouds(Occlusion):
    def __init__(self) :
        pass

    def __call__(self, img):
        pass


