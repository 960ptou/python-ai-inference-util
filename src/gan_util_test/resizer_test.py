import hypothesis
from hypothesis import given, strategies as st

import numpy as np

hypothesis.settings(max_examples=50)

from gan_util.resizer import (getModelShape, resizer)

# -------------- getModelShape ------------------ #
@given(st.tuples(st.integers(min_value=1), st.integers(min_value=1)),
       st.tuples(st.integers(min_value=1), st.integers(min_value=1)))
def test_given_shape_divisible_by_model_shape(image_shape, modelShape):
    x,y = getModelShape(image_shape, modelShape)
    xt, yt = modelShape
    assert x % xt == 0 and y % yt == 0
    
# -------------- resizer  ----------------------- #
@given(st.tuples(st.integers(min_value=1,max_value=300), st.integers(min_value=1,max_value=300)),
       st.tuples(st.integers(min_value=1,max_value=300), st.integers(min_value=1,max_value=300)),
       st.sampled_from([1,3,4]))
def test_reverse_operation_is_initial(image_shape, modelShape, channels):
    my_resizer = resizer(image_shape, modelShape, channels)
    random_image = np.random.randint(low=0,high=255,size=image_shape + (channels, ))
    np.testing.assert_array_equal(random_image,my_resizer.back(my_resizer.to(random_image)))
    
    
@given(st.tuples(st.integers(min_value=1,max_value=300), st.integers(min_value=1,max_value=300)),
       st.tuples(st.integers(min_value=1,max_value=300), st.integers(min_value=1,max_value=300)),
       st.sampled_from([1,3,4]))
def test_image_is_on_model_image(image_shape, modelShape, channels):
    my_resizer = resizer(image_shape, modelShape, channels)
    random_image = np.random.randint(low=0,high=255,size=image_shape + (channels, ))
    np.testing.assert_array_equal(random_image, my_resizer.to(random_image)[0:image_shape[0], 0:image_shape[1]])
    
@given(st.tuples(st.integers(min_value=1,max_value=300), st.integers(min_value=1,max_value=300)),
       st.tuples(st.integers(min_value=1,max_value=300), st.integers(min_value=1,max_value=300)),
       st.sampled_from([1,3,4]))
def test_resizer_shapes(image_shape, modelShape, channels):
    my_resizer = resizer(image_shape, modelShape, channels)
    random_image = np.random.randint(low=0,high=255,size=image_shape + (channels, ))
    
    x,y,chan = my_resizer.to(random_image).shape
    xt, yt = modelShape
    assert x % xt == 0 and y % yt == 0 and chan == channels