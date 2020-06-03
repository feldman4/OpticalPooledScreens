import warnings
import numpy as np
import pandas as pd
from skimage.transform import resize, rescale


def make_ball(radius):
    """Returns height profile of a sphere of given radius.
    """
    r = radius
    J, I = np.meshgrid(range(-r + 1, r), range(-r + 1, r))
    profile = r**2 - (J**2 + I**2)
    profile[profile < 0] = 0
    return profile


def slide_window(image_shape, window_shape):
    """Returns linear indices into image obtained by 
    sliding window over all valid (fully included) positions.
    """
    (h, w), (h_, w_) = image_shape, window_shape
    J, I = np.meshgrid(range(h_), range(w_))
    index_2D = J.flat[:], I.flat[:]
    # linear indices for window at the origin
    base = np.ravel_multi_index(index_2D, image_shape)
    
    # add the linear index for each valid i,j point
    h_valid, w_valid = h - h_ + 1, w - w_ + 1
    J, I = np.meshgrid(range(h_valid), range(w_valid))
    index_2D = J.flat[:], I.flat[:]
    ij_linear = np.ravel_multi_index(index_2D, image_shape)
    
    return base[None, :] + ij_linear[:, None]
    

def rolling_ball_background(image, ball):
    # image indices for comparison
    h, w = image.shape
    h_, w_ = ball.shape
    ix = slide_window(image.shape, ball.shape)

    # distance from ball to image
    x = (image.flat[ix] - ball.flat[:])
    
    # (lowest) height of the ball at each position
    distances = x.min(axis=1) #+ ball.flat[contacts]

    # height of background
    est_bkgd = distances[:, None] + ball.flat[:]

    # maximum of the background estimate for each original pixel
    # equivalent of accumarray with max
    df = pd.DataFrame({'A': est_bkgd.flatten(), 'B': ix.flatten()})

    background = (df
     .pivot_table(index='B', values='A', aggfunc='max')['A']
     .values.reshape(image.shape).astype(image.dtype))
    
    return background

def subtract_background(image, radius, shrink_factor=None, ball=None, 
    mem_cap=1e9):
    from skimage.transform import resize, rescale
    import warnings

    if ball is None:
        ball = make_ball(radius).astype(image.dtype)
        shrink_factor_, trim = imagej_heuristic(radius)

        n = int(ball.shape[0] * trim)
        i0, i1 = n, ball.shape[0] - n
        ball = ball[i0:i1, i0:i1]

        if shrink_factor is None:
            shrink_factor = shrink_factor_


    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        image_ = rescale(image, 1./shrink_factor, 
                       preserve_range=True).astype(image.dtype)

        # reduce the ball array size but keep the height the same
        ball_ = rescale(ball, 1./shrink_factor, 
               preserve_range=True).astype(ball.dtype)
        
    mem_usage = image_.size * ball_.size
    if mem_usage > mem_cap:
        error = 'not enough memory, requires {0} but capped at {1}'
        raise ValueError(error.format(mem_usage, mem_cap))
    
    background = rolling_ball_background(image_, ball_)
        
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        background = resize(background, image.shape, 
                        preserve_range=True).astype(image.dtype)
    
    # hack, shouldn't be getting points above input
    mask = background > image
    # print(mask.sum())
    background[mask] = image[mask]
    return image - background


def imagej_heuristic(radius):
    """ Copied from ImageJ "Subtract Background" command.
    """  
    if radius <= 10:
        shrink_factor = 1
        trim = 0.12 #; // trim 24% in x and y
    elif radius <= 30:
        shrink_factor = 2
        trim = 0.12 #; // trim 24% in x and y
    elif radius <= 100:
        shrink_factor = 4
        trim = 0.16 #; // trim 32% in x and y
    else:
        shrink_factor = 8
        trim = 0.20 #; // trim 40% in x and y

    return shrink_factor, trim


def split_overlap_2D(shape, width, overlap):
    """Return list of 2D index subarrays that collectively
    tile the input. The shape of a subarray is at most 
    (width + 2*overlap, width + 2*overlap).
    """
    size = shape[0] * shape[1]
    index_arr = np.arange(size).reshape(shape)
    indices = []
    i = 0
    while i < shape[0]:
        j = 0
        while j < shape[1]:
            i0 = max(i - overlap, 0)
            i1 = min(i + width + overlap, shape[0])
            j0 = max(j - overlap, 0)
            j1 = min(j + width + overlap, shape[1])
            
            indices += [index_arr[i0:i1, j0:j1]]
            
            j += width
            
        i += width
        
    return indices
            
def merge(shape, data, indices, acc_func='max', start_val=None):
    """Merge data chunks into a single array using the indices provided. 

    Example:

        indices = split_overlap_2D((10, 10), 3, 1)
        data = [np.ones_like(x) for x in indices]
        result = merge((10, 10), data, indices)

    """
    if start_val is None:
        if acc_func == 'min':
            start_val = np.iinfo(data[0].dtype).max
        else:
            start_val = 0

    arr = np.zeros(shape, dtype=data[0].dtype) + start_val
    for a, i in zip(data, indices):
        if acc_func == 'max':
            arr.flat[i] = np.maximum(arr.flat[i], a)
        elif acc_func == 'min':
            arr.flat[i] = np.minimum(arr.flat[i], a)
        elif acc_func == 'sum':
            arr.flat[i] += a
        else:
            arr.flat[i] = acc_func(arr.flat[i], a)
    return arr


def test_square():
    test = np.zeros((200, 200), dtype=np.uint16)
    test[50:150, 50:150] = 100
    return test    
