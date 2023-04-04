import numpy as np
import cv2
from PIL import Image
from typing import Literal, Callable, NamedTuple
from types import SimpleNamespace as Namespace

def getModelShape(imgShape: tuple[int, int], modelShape: tuple[int, int]) -> tuple[int, int]:
    """Function will take current image shape and give shape based on GAN model requirement

    Args:
        imgShape (tuple[int, int]): original image shape
        modelShape (tuple[int, int]): base GAN shape requirement

    Returns:
        tuple[int, int]: the result shape needed for the model
    """
    
    x0,y0 = imgShape
    xt,yt = modelShape
    return ( (x0 // xt + 1) * xt, (y0 // yt + 1) * yt)
    
    

def resizer(
    imageSize: tuple[int, int],
    modelSize: tuple[int, int],
    cchannels: Literal[1,3,4], #gray, rgb, rgba
) -> Namespace:
    targetSize = getModelShape(imageSize, modelSize)
    inputx, inputy = imageSize
    
    def makeModelImage(imgAry : np.ndarray) -> np.ndarray:
        assert imgAry.shape == imageSize + (cchannels,)
        binaryMap = np.zeros( (targetSize + (cchannels,)), dtype=np.uint8)
        if cchannels == 4: # rgba
            binaryMap[:,:,3] = 255

        binaryMap[0:inputx, 0:inputy] = imgAry
        return binaryMap
        
    def revertImage(imgAry : np.ndarray) -> np.ndarray:
        assert imgAry.shape == (targetSize + (cchannels,))
        return imgAry[0:inputx, 0:inputy]
    
    return Namespace(to=makeModelImage, back=revertImage)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    image = '/Users/weijiechen/Desktop/YouTube/imageonline-co-roundcorner.png'
    f = Image.open(image)
    fn = np.array(f)
    myr = resizer(f.size, (100,100), 4)
    plt.imshow(myr.to(fn))
    plt.show()
    
    