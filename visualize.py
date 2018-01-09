import os
import numpy as np
import cv2
import chainer
import chainer.cuda


def out_generated_image(gen, rows, cols, val_iter, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        batch = val_iter.next()
        low_batch = gen.xp.asarray([low for _, low in batch])
        img_high = np.asarray([high * 255 for high, _ in batch])
        with chainer.using_config('train', False):
            img_super = gen(low_batch)
        img_super = chainer.cuda.to_cpu(img_super.data)
        img_low = chainer.cuda.to_cpu(low_batch)

        img_super = np.asarray(np.clip(img_super * 255, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = img_super.shape
        img_super = img_super.reshape((rows, cols, 3, H, W))
        img_super = img_super.transpose(0, 3, 1, 4, 2)
        img_super = img_super.reshape((rows * H, cols * W, 3))

        img_high = img_high .astype(np.uint8)
        _, _, H, W = img_high.shape
        img_high = img_high.reshape((rows, cols, 3, H, W))
        img_high = img_high.transpose(0, 3, 1, 4, 2)
        img_high = img_high.reshape((rows * H, cols * W, 3))

        img_low = (img_low * 255).astype(np.uint8)
        _, _, H, W = img_low.shape
        img_low = img_low.reshape((rows, cols, 3, H, W))
        img_low = img_low.transpose(0, 3, 1, 4, 2)
        img_low = img_low.reshape((rows * H, cols * W, 3))

        preview_dir = '{}/preview'.format(dst)
        preview_low_path = preview_dir +\
            '/image{:0>8}_0_low.png'.format(trainer.updater.iteration)
        preview_super_path = preview_dir +\
            '/image{:0>8}_1_super.png'.format(trainer.updater.iteration)
        preview_high_path = preview_dir +\
            '/image{:0>8}_2_high.png'.format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        cv2.imwrite(preview_super_path, img_super)
        cv2.imwrite(preview_high_path, img_high)
        cv2.imwrite(preview_low_path, img_low)
    return make_image
