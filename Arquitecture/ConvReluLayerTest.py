__author__ = 'Giovanni'
import theano
import theano.tensor as T
from PIL import Image
import numpy as np

import ConvReluLayer

x = T.matrix('x')   # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels


# open Face Image 100 x 100
img = Image.open(r'E:\dev\TesisFRwithCNN\TestRS\0000117\005-l.jpg')
#DEBUG img = Image.open(r'E:\dev\TesisFRwithCNN\TestRS\convTest_white.jpg')

# dimensions are (height, width, channel)
img = np.asarray(img, dtype=theano.config.floatX) / 256
img_input = x.reshape((1, 1, img.shape[0],img.shape[1]))
#DEBUG np.savetxt('img_input.csv', img,'%5.4f' ,delimiter=',')

layer0= ConvReluLayer.ConvReluLayer(
    input=img_input,
    filter_shape=(32, 1, 3, 3),
    image_shape=(1, 1, img.shape[0],img.shape[1])
)

train_model = theano.function(
        [],
        layer0.outputJustConv,
        givens={
            x: img
            }
        )


#DEBUG valTest = train_model() #np.savetxt('out.csv', valTest[0][1],'%5.20f', delimiter=',')

print ("...")