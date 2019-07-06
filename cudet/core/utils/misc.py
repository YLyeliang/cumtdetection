from functools import partial

import numpy as np
from six.moves import map,zip

def tensor2imgs(tensor, mean=(0,0,0),std=(1,1,1),to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std,dtype=np.float32)
    imgs=[]
    for img_id in range(num_imgs):
        img = tensor[img_id,...].cpu().numpy().transpose(1,2,0)

def multi_apply(func, *args,**kwargs):

    pfunc = partial(func,**kwargs) if kwargs else func
    map_results = map(pfunc,*args)
    return tuple(map(list,zip(*map_results)))

def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """