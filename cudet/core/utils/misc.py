from functools import partial

import mtcv
import numpy as np
from six.moves import map,zip

def tensor2imgs(tensor, mean=(0,0,0),std=(1,1,1),to_rgb=True):
    """
    Transform tensor into images. (N,C,H,W) => N (H,W,C)
    """
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std,dtype=np.float32)
    imgs=[]
    for img_id in range(num_imgs):
        img = tensor[img_id,...].cpu().numpy().transpose(1,2,0)
        img = mtcv.imdenormalize(img,mean,std,to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs

def multi_apply(func, *args,**kwargs):
    # partial()该函数用于为 func 函数的部分参数指定参数值，从而得到一个转换后的函数，
    # 程序以后调用转换后的函数时，就可以少传入那些己指定值的参数
    pfunc = partial(func,**kwargs) if kwargs else func
    map_results = map(pfunc,*args)
    return tuple(map(list,zip(*map_results)))

def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """