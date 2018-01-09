import chainer
from chainer import functions as F


class SRCNNUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen = kwargs.pop('models')
        self.batch_size = kwargs.pop('batch_size')
        super(SRCNNUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        opt_gen = self.get_optimizer('gen')

        def _update(optimizer, loss):
            optimizer.target.cleargrads()
            loss.backward()
            optimizer.update()

        batch = self.get_iterator('main').next()
        img_high = [h for h, _ in batch]
        img_low = [l for _, l in batch]
        img_high = self.converter(img_high, self.device)
        img_low = self.converter(img_low, self.device)
        gen = self.gen
        img_super = gen(img_low)

        # mse
        mse_loss = F.mean_squared_error(img_high, img_super)

        # only update by mse
        loss = mse_loss

        # update gen
        _update(opt_gen, loss)
        chainer.report({'Gen/loss': loss, 'mse_loss': mse_loss})
