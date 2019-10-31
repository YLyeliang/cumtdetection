from .xml_style import XMLDataset


class LeakageDataset(XMLDataset):

    CLASSES = ('water',)

    def __init__(self, **kwargs):
        super(LeakageDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')
