import tensorflow as tf
import gin


def MSE_loss(gt, pred, mask):

    # (X- X')^2
    # loss = tf.math.square(tf.subtract(gt, pred))
    loss = tf.subtract(gt, pred)
    loss = tf.multiply(mask, loss)
    loss = tf.divide(loss, loss.shape[1]*loss.shape[2])

    return tf.reduce_mean(loss)


@gin.configurable
def weighted_MSE_with_gals(gt, pred, bckg_weight, galaxy_weight):
    ''' put less weight on neg pixels'''

    ''' ##### Clumps loss  #####'''
    ''' ## bckg loss ##'''

    c_bckg_mask = tf.where(gt[..., 0] == 0, 1., 0.)
    c_bckg_loss = MSE_loss(gt[..., 0], pred[..., 0], c_bckg_mask)

    ''' ## positive pixels loss ##'''
    clumps_mask = tf.where(gt[..., 0] > 0, 1., 0.)
    c_loss = MSE_loss(gt[..., 0], pred[..., 0], clumps_mask)

    clumps_loss = bckg_weight*c_bckg_loss + c_loss

    ''' ##### galaxy loss  #####'''
    ''' ## bckg loss ##'''
    g_bckg_mask = tf.where(gt[..., 1] == 0, 1., .0)
    g_bckg_loss = MSE_loss(gt[..., 1], pred[..., 1], g_bckg_mask)

    ''' ## positive pixels loss ##'''
    galaxy_mask = tf.where(gt[..., 1] > 0, 1., 0.)
    g_loss = MSE_loss(gt[..., 1], pred[..., 1], galaxy_mask)

    galaxy_loss = bckg_weight * g_bckg_loss + g_loss

    return [clumps_loss, galaxy_loss,
            clumps_loss + galaxy_weight * galaxy_loss]


def weighted_MSE_clumps_only(gt, pred, bckg_weight, galaxy_weight=None):
    ''' put less weight on neg pixels'''

    ''' ##### Clumps loss  #####'''
    mask = tf.ones_like(gt[..., 0])
    mask = tf.where(gt[..., 0] == 0, 1., 0.)
    # L = mask * |X-X'|^2'''
    c_loss = tf.math.abs(tf.math.square(tf.multiply(mask,
                                        tf.subtract(gt[..., 0],
                                                    pred[..., 0]))))

    # L = L / N_pix'''
    c_loss = tf.divide(c_loss, gt.shape[1]*gt.shape[2])

    # return list to be able to take -1 regardless of the choice of loss
    return [c_loss] 
