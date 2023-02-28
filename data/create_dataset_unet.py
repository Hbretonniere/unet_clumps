import galsim
import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.table import Table
import sys
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
sys.path.append('../../..')
from dataset_utils import add_clumps_to_field


cat = Table.read('./raw/gal_DEC_2022_input_ok.cat', format='ascii')
cat_stars = Table.read('./raw/ptsrc_DEC_2022_input.cat', format='ascii')
cw_RAs = np.concatenate((np.array(cat['x_or_RA']),
                         np.array(cat_stars['x_or_RA'])))
cw_Decs = np.concatenate((np.array(cat['y_or_Dec']),
                          np.array(cat_stars['y_or_Dec'])))


# psf_file = './raw/nircam_nrcb5_f444w_clear_fovp61.fits'
# psf = fits.open(psf_file)[0].data
# hdu = fits.PrimaryHDU(psf[0])
# hdulist = fits.HDUList([hdu])
# hdulist.writeto('raw/NIRCam_f444w.fits', overwrite=True)
psf = galsim.InterpolatedImage('raw/NIRCam_f444w.fits',
                               flux=1,
                               scale=0.031)

hdu_list = fits.open('./raw/mosaic_nircam_f444w_COSMOS-Web_i2d.fits',
                     ignore_missing_simple=True)
cw = hdu_list[1].data
w = wcs.WCS(hdu_list[1].header, naxis=2)


cw_clumps = fits.open('./raw/mosaic_nircam_f444w_COSMOS-Web_i2d.fits')[1].data
n_gals = 6000

nb_fields = 4
i = 0
for field in range(nb_fields):
    (cw_clumps,
     clump_cat,
     seg) = add_clumps_to_field(cw, cw_clumps, cat,
                                cw_RAs, cw_Decs, w,
                                psf,
                                n_gals,
                                mag_max=26,
                                mag_min=17,
                                hlr_min=8,
                                sigma_segmap=6,
                                show=20)

    columns = []
    for key in clump_cat.keys():
        print(key)
        columns.append(fits.Column(name=key, format='D', array=clump_cat[key]))

    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.writeto(f'clump_catalogue_{field}.fits', overwrite=True)

    hdu = fits.PrimaryHDU(cw_clumps)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(f'./clumpsy_cosmos_web_{field}.fits', overwrite=True)

    hdu = fits.PrimaryHDU(seg)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(f'./centroid_segmap_clumpsy_cosmos_web_{field}.fits',
                    overwrite=True)

    '''
    #################################
              cut the stamps
    #################################'''

    ccw = fits.open(f'clumpsy_cosmos_web_{field}.fits')[0].data
    segmap = fits.open(f'centroid_segmap_clumpsy_cosmos_web_{field}.fits')[0].data

    sx, sy = ccw.shape
    x_start = np.random.randint(0, 128)
    y_start = np.random.randint(0, 128)
    stamp_size = 128
    show = 20
    fig, ax = plt.subplots(show, 2, figsize=(10, 30))
    interval = ZScaleInterval()
    margin = 30
    for ix in range((sx-x_start)//stamp_size):
        for iy in range((sy-y_start)//stamp_size):

            xs, xe = x_start+ix*stamp_size, x_start+ix*stamp_size+stamp_size
            ys, ye = y_start+iy*stamp_size, y_start+iy*stamp_size+stamp_size
            stamp = ccw[xs:xe, ys:ye]
            seg = segmap[xs:xe, ys:ye]
            if np.max(seg[margin:-margin, margin:-margin, 0]) == 0:
                continue
            else:
                a = interval.get_limits(stamp)
                hdu = fits.PrimaryHDU(stamp)
                hdulist = fits.HDUList([hdu])
                hdulist.writeto(f'./training_images/train_img_{i}.fits',
                                overwrite=True)

                hdu = fits.PrimaryHDU(seg)
                hdulist = fits.HDUList([hdu])
                hdulist.writeto(f'./training_segmaps/train_seg_{i}.fits',
                                overwrite=True)
                if i < show:
                    ax[i, 0].imshow(stamp, vmin=a[0], vmax=a[1], cmap='bone')
                    ax[i, 1].imshow(seg[..., 0])
                i += 1

plt.savefig('training_set_no-preprocess.png')
