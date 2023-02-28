import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
import numpy as np


def plot_box_from_center(ax, x, y, s_stamp, s_box, s_star=100, center=True):

    if center:
        ax.scatter(s_stamp//2+y, s_stamp//2+x, marker='*',
                   s=s_star, color='red',
                   zorder=2)

    ax.plot([s_stamp//2+y-s_box, s_stamp//2+y+s_box],
            [s_stamp//2+x-s_box, s_stamp//2+x-s_box],
            c='red')

    ax.plot([s_stamp//2+y-s_box, s_stamp//2+y+s_box],
            [s_stamp//2+x+s_box, s_stamp//2+x+s_box],
            c='red')

    ax.plot([s_stamp//2+y-s_box, s_stamp//2+y-s_box],
            [s_stamp//2+x-s_box, s_stamp//2+x+s_box],
            c='red')

    ax.plot([s_stamp//2+y+s_box, s_stamp//2+y+s_box],
            [s_stamp//2+x-s_box, s_stamp//2+x+s_box],
            c='red')

    return ax


def plot_box_from_start(ax, x, y, s_x=5, s_y=5, s_star=25, center=True):

    if center:
        ax.scatter(y, x, marker='*',
                   s=s_star, color='red',
                   zorder=2)

    ax.plot([y-s_y, y+s_y],
            [x-s_x,  x-s_x],
            c='red')

    ax.plot([y-s_y,  y+s_y],
            [x+s_x,  x+s_x],
            c='red')

    ax.plot([y-s_y,  y-s_y],
            [x-s_x,  x+s_x],
            c='red')

    ax.plot([y+s_y,  y+s_y],
            [x-s_x,  x+s_x],
            c='red')

    return ax


def crop(array, x, y, crop_size):
    return array[x-crop_size//2:x+crop_size//2,
                 y-crop_size//2:y+crop_size//2]


def zoom(array, tot_size, crop_size):
    return array[tot_size//2-crop_size//2:tot_size//2+crop_size//2,
                 tot_size//2-crop_size//2:tot_size//2+crop_size//2]


def inspect_clumps_creation(original_field, clumpy_field,
                            galaxy_stamp, clumpy_stamp,
                            gal_x, gal_y,
                            gal_hlr, n_clumps,
                            clump_cat,
                            first_clump_id,
                            gal_flux,
                            px_scale,
                            segmap=None):

    ''' Plot large fields, with or without new gal'''
    fig, ax = plt.subplots(3, 2, figsize=(20, 30))

    plot_s_field = 400

    cw_zoom = crop(original_field, gal_x, gal_y, plot_s_field)
    cw_clumps_zoom = crop(clumpy_field, gal_x, gal_y, plot_s_field)
    interval = ZScaleInterval()
    a = interval.get_limits(cw_zoom)

    ax[0, 0].imshow(cw_zoom, cmap='bone', vmin=a[0], vmax=a[1])#,
                #     origin='lower')
    ax[0, 0].set_title('original cosmos_web simulation')

    a = interval.get_limits(cw_clumps_zoom)
    ax[0, 1].imshow(cw_clumps_zoom, cmap='bone',
                    vmin=a[0], vmax=a[1]) #, origin='lower')
    ax[0, 1].set_title('galaxy with clumps added')

    ''' Plot the galaxy and galaxy + clumps stamps'''
    size_stamp = np.shape(galaxy_stamp)[0]
    s_zoom = 2*int(6*gal_hlr / px_scale)
    zoom_stamp_galaxy = zoom(galaxy_stamp, size_stamp, s_zoom)
    segmap_zoom = crop(segmap, gal_x, gal_y, s_zoom)

    ax[1, 0].imshow(zoom_stamp_galaxy, cmap='flag')#, origin='lower')
    ax[1, 0].set_title('added galaxy (no clumps)')

    zoom_stamp_gal_clump = zoom(clumpy_stamp, 3*size_stamp, s_zoom)
    ax[1, 1].imshow(zoom_stamp_gal_clump, cmap='flag')#, origin='lower')
    ax[1, 1].set_title('added galaxy (with clumps)')

    clumps_flux = 0
    s_box = 10

#     ''' Plot the diff and the seg_map'''
#     clumps_only = zoom_stamp_gal_clump - zoom_stamp_galaxy
#     ax[2, 0].imshow(clumps_only +
#                     np.random.normal(0, 0.0074, (s_zoom, s_zoom)),
#                     cmap='bone')#, origin='lower')

#     for i in range(n_clumps):
#         # print(first_clump_id+i-1)
#         clump_y = clump_cat['y_off'][first_clump_id+i]
#         clump_x = clump_cat['x_off'][first_clump_id+i]
#         clumps_flux += clump_cat['flux'][first_clump_id+i]
#         plot_box_from_center(ax[2, 0], clump_x, clump_y, s_zoom, s_box)

#     if segmap:

    ax[2, 0].imshow(segmap_zoom[:, :, 0], cmap='bone')
    ax[2, 1].imshow(segmap_zoom[:, :, 1], cmap='bone')
#     ax[2, 1].imshow(segmap_zoom, cmap='bone')

    title = f'{n_clumps} clumps, for a total of' + \
            f'{np.round(clumps_flux, 4)},' + \
            f'i.e. {np.round(clumps_flux/gal_flux * 100, 1)}' +\
            '% of the total flux'
    ax[1, 1].set_title(title)

    ax[2, 0].set_title('clumps only with noise')

    return fig
