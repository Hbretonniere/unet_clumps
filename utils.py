import tensorflow as tf
import gin
import numpy as np


def tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


@gin.configurable()
def ds9_scaling(img, scaling_factor, offset=True):
    a = tf.cast(scaling_factor, tf.float64)
    if offset:
        img -= np.min(img)
    img = tf_log10(a*img + 1)/tf_log10(a + 1)
    nan_free_img = tf.where(tf.math.is_nan(img), tf.cast(0, tf.float64), img)
    return nan_free_img


initial_lr = 0.001  # Initial learning rate
lr_decay = 0.95  # Decay rate for learning rate
lr_schedule = {
    'layer1': 1.0,  # learning rate for layer1
    'default': 0.1  # learning rate for all other weights
}


# class Custom_lr_decay(tf.keras.optimizers.schedules.LearningRateSchedule):

#     def __init__(self, initial_lr, lr_decay, lr_schedule):
#         super(Custom_lr_decay, self).__init__()
#         self.initial_lr = initial_lr
#         self.lr_decay = lr_decay
#         self.lr_schedule = lr_schedule

#     def __call__(self, step):
#         lr = self.initial_lr
#         for layer_name in self.lr_schedule.keys():
#             if layer_name in step.name:
#                 lr *= self.lr_schedule[layer_name]
#                 break
#         else:
#             lr *= self.lr_schedule.get('default', 1.0)
#         return lr / tf.math.pow(self.lr_decay, tf.cast(step, tf.float32))


# optimizer = tf.keras.optimizers.Adam(
#     learning_rate=CustomSchedule(initial_lr, lr_decay, lr_schedule))
