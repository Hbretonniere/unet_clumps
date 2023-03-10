import os
os.environ['PATH'] = '/home/hbretonn/morpheus/big-morpheus_JADES/morpheus_env/lib/python3.8/:' + os.environ['PATH']
from architectures.unet import UNet
import numpy as np
from astropy.io import fits
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as tfk
import gin
import argparse
from utils import ds9_scaling
import glob
import comet_utils


def config_str_to_dict(config_str: str) -> dict:
    """Converts a Gin.config_str() to a dict for logging with comet.ml"""

    predicate = lambda x: len(x) > 0 and not (
        x.startswith("#") or x.startswith("import")
    )
    to_kv = lambda x: [line.strip() for line in x.split("=")]

    lines = config_str.splitlines()
    return {k: v for k, v in map(to_kv, filter(predicate, lines))}


@gin.configurable()
def train_model(stamp_size: int,
                n_classes: int,
                N_colors: int,
                channels: list,
                block_size: int,
                batch_size: int,
                train_slice: float,
                eval_slice: float,
                training_epochs: int,
                log_metric_frequency: int,
                plot_frequency: int,
                save_weights_epochs: int,
                grad_clip_value: float,
                learning_rate_schedule: str,
                init_lr: float,
                band,
                comet_experiment_key,
                comet_project_name,
                model_code_file,
                comet_disabled,
                last_activation,
                ):

    experiment = comet_utils.setup_experiment(
        comet_experiment_key,
        comet_project_name,
        config_str_to_dict(gin.config_str(max_line_length=int(1e5))),
        model_code_file,
        comet_disabled,
    )

    '''reading training data'''

    img_path = f'./data/training_images_{band}/'
    seg_path = f'./data/training_segmaps_{band}/'
    imList = glob.glob(f"{img_path}*.fits")
    checkpoint_path = f'models/{experiment.get_key()}/'
    N = len(imList)
    imgs = np.zeros((N, stamp_size, stamp_size, N_colors)).astype('float32')
    segs = np.zeros((N, stamp_size, stamp_size, 2)).astype('float32')

    start = 0 #1005
    for i, idx in enumerate(np.arange(start, start+N)):
        # imgs[i, :, :, 0] = ds9_scaling(fits.open(f'{img_path}train_img_{idx}.fits')[0].data)
        imgs[i, :, :, 0] = fits.open(f'{img_path}train_img_{idx}.fits')[0].data
        segs[i, :, :, :] = fits.open(f'{seg_path}train_seg_{idx}.fits')[0].data

    # fig, ax = plt.subplots(20, 2, figsize=(10, 40))
    # for i in range(20):
    #     ax[i, 0].imshow(imgs[i, :, :, 0])
    #     ax[i, 1].imshow(segs[i, :, :, 0])

    # plt.savefig('inputs.png')

    train_slice = int(train_slice * imgs.shape[0])
    eval_slice = int(eval_slice * imgs.shape[0])
    steps_per_epoch = int(train_slice / batch_size)

    train_data = tf.data.Dataset.from_tensor_slices((imgs[0:train_slice],
                                                     segs[0:train_slice]))
    train_data = train_data.shuffle(100000, reshuffle_each_iteration=True)
    train_data = train_data.batch(batch_size)
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
    # eval_data = tf.data.Dataset.from_tensor_slices(
    # (imgs[train_slice:eval_slice], segs[train_slice:eval_slice]))
    # .shuffle(100000, reshuffle_each_iteration=True)
    # .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # test_images = imgs[-500:, :, :]
    # test_segs = segs[-500:, :, :]

    del imgs
    del segs

    if learning_rate_schedule == 'exp_decay':
        schedule = tf.keras.optimizers.schedules.ExponentialDecay
        lr_schedule1 = schedule(init_lr*10000,
                                decay_steps=steps_per_epoch*training_epochs,
                                decay_rate=0.02)
        lr_schedule2 = schedule(init_lr,
                                decay_steps=steps_per_epoch*training_epochs,
                                decay_rate=0.02)

        optimizers = [tf.keras.optimizers.Adam(lr_schedule1),
                      tf.keras.optimizers.Adam(lr_schedule2,
                                               clipvalue=grad_clip_value)]

        # optimizer = tf.keras.optimizers.Adam(lr_schedule,
                                            #  clipvalue=grad_clip_value)
    elif learning_rate_schedule == 'cst':
        optimizers = [tf.keras.optimizers.Adam(init_lr*1000),
                      tf.keras.optimizers.Adam(init_lr)]

    ''' Create the model '''
    model = UNet((stamp_size, stamp_size, 1),
                 channels,
                 n_classes,
                 block_size,
                 last_activation,
                 checkpoint_path,
                 optimizers,
                 experiment)

    model.print_models('models_summary/')

    ''' Train '''

    model.train(train_data,
                training_epochs,
                log_metric_frequency,
                plot_frequency,
                save_weights_epochs)


def main(config_file: str) -> None:
    gin.parse_config_file(config_file)
    train_model()  # pylint: disable=no-value-for-parameter


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Trainer")
    parser.add_argument("config", help="Gin config file with model params.")
    main(parser.parse_args().config)
