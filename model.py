import chainer
import chainer.functions as F
import chainer.links as L


class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(in_channels=3, out_channels=64, ksize=9, pad=4)
            self.c2 = L.Convolution2D(in_channels=64, out_channels=32, ksize=1)
            self.c3 = L.Convolution2D(in_channels=32, out_channels=3, ksize=5, pad=2)

    def __call__(self, x):
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        h = self.c3(h)
        return h









