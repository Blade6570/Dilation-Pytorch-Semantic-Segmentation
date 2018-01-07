import numpy as np

# Configurations for different datasets
CONFIG = {
    'cityscapes': {
        'classes': 19,
        'weights_file': 'data/pretrained_dilation_cityscapes.pickle',
        'input_shape': (1396, 2420,3),
        'output_shape': (1024, 1024, 3),
        'mean_pixel': (72.39, 82.91, 73.16),
        'palette': np.array([[128, 64, 128],
                            [244, 35, 232],
                            [70, 70, 70],
                            [102, 102, 156],
                            [190, 153, 153],
                            [153, 153, 153],
                            [250, 170, 30],
                            [220, 220, 0],
                            [107, 142, 35],
                            [152, 251, 152],
                            [70, 130, 180],
                            [220, 20, 60],
                            [255, 0, 0],
                            [0, 0, 142],
                            [0, 0, 70],
                            [0, 60, 100],
                            [0, 80, 100],
                            [0, 0, 230],
                            [119, 11, 32]], dtype='uint8'),
        'zoom': 1,
        'conv_margin': 186
    }

}
