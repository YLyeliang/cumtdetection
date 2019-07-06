import torch


class AnchorGenerator(object):

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size  # 4 8 16 32 64
        h = self.base_size  # e.g 4
        # calculate center of x & y
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)  # 1.5
            y_ctr = 0.5 * (h - 1)  # 1.5
        else:
            x_ctr, y_ctr = self.ctr

        # Aspect ratio of anchor box    # width / height
        h_ratios = torch.sqrt(self.ratios)  # 0.5 1 2
        w_ratios = 1 / h_ratios  # 2 1 0.5
        if self.scale_major:  # w * ratio * scale ,In pytorch tensor, if 1D,add None will get 2D e.g[N,1]
            # scales [8,16,32]
            # img size :(1000,600), after conv, (500,300) (250,150)     anchor size 4,
            # 每个scale有三个ratio，三个scale相当于9个anchor box.
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)  # 矩阵乘法 4 *(3,1) *(1,3) => (3,3) size.
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)  # 16 32 64 32 64 128 64 128 256
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)

        # xmin, ymin,xmax,ymax
        base_anchors = torch.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            dim=-1).round()

        return base_anchors

    def _meshgrid(self,x,y,row_major=True):
        # make meshgrid.
        xx= x.repeat(len(y))        # e.g 3*3 grid.   x:range(0,3)  xx= 0 1 2 0 1 2 0 1 2
        yy=y.view(-1,1).repeat(1,len(x)).view(-1)   # yy = 0 0 0 1 1 1 2 2 2
        if row_major:
            return xx,yy
        else:
            return yy,xx

    def grid_anchors(self,featmap_size,stride=16,device='cuda'):
        base_anchors = self.base_anchors.to(device)

        feat_h, feat_w = featmap_size  # e.g  h:250, w:150
        shift_x = torch.arange(0,feat_w,device=device)*stride   #
        shift_y = torch.arange(0,feat_h,device=device) *stride
        shift_xx, shift_yy = self._meshgrid(shift_x,shift_y)
        shifts = torch.stack([shift_xx,shift_yy,shift_xx,shift_yy],dim=-1)  # xmin,ymin,xmax,ymax shift.
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        # simply saying,  H*W*No_anchor=all anchors
        all_anchors = base_anchors[None, :,:] + shifts[:,None,:]
        all_anchors = all_anchors.view(-1,4)
        # first A rows correspond to A anchors of (0,0) in feature map.
        return all_anchors

    def valid_flags(self,featmap_size, valid_size, device='cuda'):
        feat_h, feat_w =featmap_size
        valid_h, valid_w =valid_size
        assert valid_h <=feat_h and valid_w <=feat_w
        valid_x = torch.zeros(feat_w,dtype=torch.uint8,device=device)
        valid_y = torch.zeros(feat_h,dtype=torch.uint8,device=device)
        valid_x[:valid_w] =1
        valid_y[:valid_h]=1
        valid_xx, valid_yy = self._meshgrid(valid_x,valid_y)
        # all valid region are set to 1.
        valid = valid_xx & valid_yy
        valid = valid[:,None].expand(
            valid.size(0),self.num_base_anchors).contiguous().view(-1)  # (valid size,num anchor).view(-1) valid:(validH*validW*numAnchor)
        return valid





