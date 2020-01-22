# Created by raouf at 19/01/2020

from data.utils import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class  Occlusion(object):

    def __init__(self):
        raise NotImplementedError

    def __call__(self, video):
        """
        :param video : Tensor ( nb_frames x C x H x W )
        :return: occluded video : Tensor ( nb_frames x C x H x W )
        """
        raise NotImplementedError

class RainDrops(Occlusion):
    """
    Simulate rain
    """
    def __init__(self, number=None, positions=None, width=1.0/64, height=None, speed=None, output_size=(64, 64) , mask_code = -100 ):

        """
        :param number     : int  => number of raindrops ( vertical lines )
        :param positions  : array_like List , ndarray , ... of Floats in 0 .. 1 with shape (number,2) => each line contains coordinates (x,y)
        :param width      : Float => a fixed scaling number in 0 .. 1 representing the width of different vertical lines
        :param height     : array_like List , ndarray , ... of Floats in 0 .. 1 with shape (number,)
        :param speed      : array_like List , ndarray , ... of Floats in 0 .. 1 with shape (number,)
        :param output_size: tuple of 2 (int,int)  ==> size of the ouput frames
        :param mask_code  : Int representing mask code  ( choose negative ints )
        """
        # if data is not given create random stuff
        if ( number is None ) :
            number = np.random.randint(low=0,high=output_size[1])
        if ( positions is None ) :
            positions = np.random.rand(number,2)
        if ( height is None ) :
            height = np.random.rand(number)
        if ( speed is None ) :
            speed = np.random.rand(number)

        # convert [0,1] scale to pixels language
        positions = torch.round(torch.tensor(positions) * output_size[0]).int()
        width     = torch.round(torch.tensor(width)     * output_size[1]).int().item()
        height    = torch.round(torch.tensor(height)    * output_size[0]).int()
        speed     = torch.round(torch.tensor(speed)     * output_size[0]).int()

        self.number    = number
        self.positions = positions
        self.width     = width
        self.height    = height
        self.speed     = speed
        self.mask_code = mask_code
        self.output_size=output_size

    def __call__(self, video):
        frame_height, frame_width = video.shape[2], video.shape[3]
        video_tensor = []

        nbr_frames = 0
        for frame in video :

            #frame_copy = frame.clone()

            for i in range((self.positions).shape[0]) :

                pos = self.positions[i]
                #mask x
                mask_x = torch.arange(start=int(pos[0]), end=int(pos[0] + self.height[i])).cpu()
                #mask y
                mask_y = torch.arange(start=int(pos[1]), end=int(pos[1] + self.width)).cpu()
                grid_x, grid_y = torch.meshgrid(mask_x, mask_y)
                frame = frame.cpu()
                frame[:, grid_x%self.output_size[0] , grid_y%self.output_size[1]  ] = self.mask_code
                frame.to(device)

            # append the frame
            video_tensor.append(frame.unsqueeze(0))

            # next starting X position = current X position + speed
            self.positions[:, 0] += self.speed

            nbr_frames += 1

        return torch.cat(video_tensor)

class RemovePixels(Occlusion):
    """
    Simulates severe damages on vintage films
    """
    def __init__(self, threshold=0.1 , mask_code = -100) :
        self.threshold = threshold
        self.mask_code = mask_code

    def __call__(self, video ):

        video_copy = video.clone()
        # uniform random sampling for each pixel through the whole video sequence
        p = torch.rand(video_copy.shape)
        # affect mask_code  to the selected pixels
        video_copy[p <= self.threshold] = self.mask_code

        return video_copy



class MovingBar(Occlusion):

    """
    Simulates moving obstacle
    """
    def __init__(self, position=None , width=None , height=None, speed=None , output_size=(64, 64), mask_code=-100):

        # if data is not given create random stuff
        if(position is None ) :
            position = np.random.rand(2)
        if ( width is None ) :
            width = np.random.rand()
        if ( height is None ) :
            height = np.random.rand()
        if ( speed is None ) :
            speed = np.random.rand()

        # convert [0,1] scale to pixels language
        position = torch.round(torch.tensor(position) * output_size[0]).int()
        width    = torch.round(torch.tensor(width)    * output_size[1]).int().item()
        height   = torch.round(torch.tensor(height)   * output_size[0]).int().item()
        speed    = torch.round(torch.tensor(speed)    * output_size[0]).int().item()

        self.position = position
        self.width = width
        self.height = height
        self.speed = speed
        self.mask_code = mask_code
        self.output_size = output_size

    def __call__(self, video ):

        video_tensor = []

        nbr_frames = 0
        for frame in video :

            frame_copy = frame.clone()
            pos = self.position

            #mask x
            mask_x = torch.arange(start=int(pos[0]), end=int(pos[0] + self.height))
            #mask y
            mask_y = torch.arange(start=int(pos[1]), end=int(pos[1] + self.width))

            grid_x, grid_y = torch.meshgrid(mask_x,mask_y)
            frame_copy[:, grid_x%self.output_size[0] , grid_y%self.output_size[1] ] = self.mask_code

            # append the frame
            video_tensor.append(frame_copy.unsqueeze(0))

            # next starting X position = current X position + speed
            self.position[1] += self.speed

            nbr_frames += 1

        return torch.cat(video_tensor)




class Clouds(Occlusion):
    def __init__(self) :
        pass

    def __call__(self, img):
        pass


