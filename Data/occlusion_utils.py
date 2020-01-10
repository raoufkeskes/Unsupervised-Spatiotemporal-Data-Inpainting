import numpy as np
import functools
import operator


img_size       = (64,64)


# flatten a list of list quickly 
def functools_reduce_iconcat(a):
    return functools.reduce(operator.iconcat, a, [])

def raindrops ( img,  number, positions , width, height ) :
  
    # convert [0,1] scale to pixels language 
    # np.intp : Integer used for indexing
    positions = np.round(positions*img_size[0]).astype(np.intp)
    width     = np.round(width*img_size[0]).astype(np.intp)
    height    = np.round(height*img_size[1]).astype(np.intp)
    
    # create masks
    #x
    extended = [ np.arange(h).tolist()  for h in height ]
    extended = np.array ( functools_reduce_iconcat ( extended ) )  
    mask_x   = np.repeat ( positions[:,0].reshape(1,-1) , height )
    mask_x  += extended
    mask_x   = np.repeat( mask_x , width )

    #y
    extended = [ ( np.repeat(np.arange(width),h) ).reshape(width,-1).T.reshape(-1).tolist()  for h in height ]
    extended = np.array ( functools_reduce_iconcat ( extended ) )  
    mask_y   = np.repeat ( positions[:,1].reshape(1,-1) , height*width )
    mask_y  += extended

    # put the mask on the image
    img[ mask_x % img_size[0], mask_y % img_size[1] ] = 255

    # rgb to bgr  for opencv
    return img[:, :, ::-1]


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