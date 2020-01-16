import numpy as np



img_size       = (64,64)




def raindrops ( img ) :
  



def remove_pixel(img, threshold):
    # uniform random sampling for each pixel
    p = np.random.rand(*img_size)
    # affect c=0 to the selected pixels
    img[p <= threshold] = 0

    # rgb to bgr  for opencv
    return img[:, :, ::-1]

def moving_vertical_bar ( img ,
                          position  = [0.1,0.1] ,
                          width     = 0.2 ,
                          height    = 0.4 ) :

    # convert [0,1] scale to pixels language
    # np.intp : Integer used for indexing
    position = np.round(position*img_size[0]).astype(np.intp)
    width    = np.round(width*img_size[0]).astype(np.intp)
    height   = np.round(height*img_size[1]).astype(np.intp)

    img [ position[0] : (position[0]+height) , position[1] : (position[1]+width) ] = 255

    # horizontal overlapping
    if ( position[1]+width > img_size[1] ) :
        img [ position[0] : (position[0]+height) ,  : (position[1]+width)%img_size[1] ] = 255

    # rgb to bgr  for opencv
    return img[:, :, ::-1]