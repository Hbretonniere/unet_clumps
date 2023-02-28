import tensorflow as tf


def tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def ds9_scaling(img, a=tf.cast(5, tf.float64), offset=0.):
    img = tf_log10(a*img + 1)/tf_log10(a + 1)  # - offset
    nan_free_img = tf.where(tf.math.is_nan(img), tf.cast(0, tf.float64), img)
    return nan_free_img