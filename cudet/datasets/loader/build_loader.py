from functools import partial

from torch.utils.data import DataLoader
from mmcv.runner import get_dist_info
from mmcv.parallel import collate

from .sampler import GroupSampler, DistributedGroupSampler,DistributedSampler



def build_dataloader(dataset,
                     imgs_per_gpu,
                     worker_per_gpu,
                     num_gpus=1,
                     dist=True,
                     **kwargs):
    shuffle = kwargs.get('shuffle',True)    # get shuffle parameters, if None, return default:True
    if dist:
        rank, world_size = get_dist_info()
        if shuffle:
            sampler = DistributedGroupSampler(dataset,imgs_per_gpu,world_size,rank)
        else:
            sampler = DistributedSampler(dataset,imgs_per_gpu,world_size,rank,shuffle=False)
        batch_size = imgs_per_gpu
        num_workers = worker_per_gpu
    else:
        sampler = GroupSampler(dataset,imgs_per_gpu) if shuffle else None # generate samples from dataset class
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * worker_per_gpu

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate,samplers_per_gpu=imgs_per_gpu),
        pin_memory=False,
        **kwargs)
    return data_loader