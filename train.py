import argparse
import chainer
from chainer import training, serializers
from chainer.training import extensions
from model import Generator
from dataset import PreprocessedDataset
from updater import SRCNNUpdater
from visualize import out_generated_image



def main():
    parser = argparse.ArgumentParser(description='Chainer: SRCNN-Chainer')
    parser.add_argument('--batch_size', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=2000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--snapshot_interval', type=int, default=100,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=10,
                        help='Interval of displaying log to console')
    parser.add_argument('--train_dir', type=str, default='/data/chen/DIV2K_train_HR/*.png',
                        help='train image dir')
    parser.add_argument('--val_dir', type=str, default='/data/chen/DIV2K_valid_HR/DIV2K_valid_HR/*.png',
                        help='val image dir')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batch_size))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up models
    gen = Generator()

    # start from epoch 4000
    # serializers.load_npz('gen_iter_4000.npz', gen)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.00001):
        optimizer = chainer.optimizers.Adam(alpha=alpha)
        optimizer.setup(model)
        # optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer

    opt_gen = make_optimizer(gen)

    train = PreprocessedDataset(args.train_dir)
    val = PreprocessedDataset(args.val_dir)

    train_iter = chainer.iterators.SerialIterator(train, args.batch_size, repeat=True, shuffle=True,)
    val_iter = chainer.iterators.SerialIterator(val, 1, repeat=True, shuffle=True)

    # Set up a trainer
    updater = SRCNNUpdater(
        models=gen,
        iterator=train_iter,
        optimizer={'gen': opt_gen},
        device=args.gpu,
        batch_size=args.batch_size,
    )

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(extensions.observe_lr(optimizer_name='gen', observation_key='lr'), trigger=display_interval)
    trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(gen, 'gen_iter_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'lr', 'Gen/loss', 'mse_loss']),
                   trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(out_generated_image(gen, 1, 1, val_iter, args.out), trigger=(10000, 'iteration'))

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
