import numpy as np
import galsim
from plot_utils import inspect_clumps_creation, plot_box_from_start
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from scipy.stats import multivariate_normal
from scipy.signal import convolve as scipy_convolve


def mag2flux(mag, zp=28.02):
    return 10**(-0.4*(mag-zp))


def flux2mag(flux, zp=28.02):
    return zp - 2.5*np.log10(flux)


def create_sub_cat(cat, max_mag, min_mag, min_hlr, px_scale=0.031):
    '''
    min_hlr: pixels
    '''
    indices = np.where((cat['nircam_f444w_clear_magnitude'] < max_mag) &
                       (cat['nircam_f444w_clear_magnitude'] > min_mag) &
                       (cat['radius']/px_scale > min_hlr))
    print(f'There is {len(indices[0])} galaxies in the sub cat')
    return cat[indices]


def assign_clump_offset(gal_hlr,
                        pos_interval,
                        px_scale):

    ''' Compute clump offset'''

    min_dist = pos_interval[0] * gal_hlr / px_scale
    max_dist = pos_interval[1] * gal_hlr / px_scale
    radius = np.random.uniform(min_dist, max_dist)
    theta = np.random.randint(0, 2*np.pi)
    clump_x = radius * np.cos(theta)
    clump_y = radius * np.sin(theta)

    return int(clump_x), int(clump_y)


def sim_clump(gal_flux,
              clumps_flux,
              stamp_size,
              px_scale,
              flux_interval,
              flux_tot_max,
              psf,
              hlr_interval=[0.5, 1]):

    stamp_clump = galsim.ImageF(stamp_size, stamp_size, scale=px_scale)
    clump_hlr = np.random.uniform(hlr_interval[0]*px_scale,
                                  hlr_interval[1]*px_scale)
    clump = galsim.Exponential(clump_hlr)

    if 0.45 * gal_flux - clumps_flux > flux_interval[1]*(gal_flux+clumps_flux):
        flux_max = flux_interval[1] * gal_flux
    elif flux_interval[0] * gal_flux < \
        flux_tot_max * gal_flux - clumps_flux < \
            flux_interval[1] * (gal_flux+clumps_flux):
        flux_max = flux_tot_max * gal_flux - clumps_flux
    else:
        flux_max = 0

    if flux_max == 0:
        clump_flux = 0
    else:
        clump_flux = np.random.uniform(0.06 * gal_flux, flux_max)

    # clump = galsim.Convolve(clump, psf)
    clump = clump.withFlux(clump_flux)
    clump.drawImage(stamp_clump, method='no_pixel')

    return stamp_clump, clump_flux


def find_empty_coords(busy_RAs, busy_Decs, n_gals, wcs,
                      RA_size=0.0008, Dec_size=0.0008):

    n = 0
    count = 0
    new_RAs = []
    new_Decs = []

    while n < n_gals:
        RA = np.random.uniform(149.87, 150)
        Dec = np.random.uniform(2.3, 2.45)
        gal_y, gal_x = wcs.wcs_world2pix(RA, Dec, 1)
        if gal_x > 6500 or gal_x < 500 or \
           gal_y < 500 or gal_y > 11000:
            continue
        cond_empty = len(np.where((busy_RAs > RA - RA_size) &
                                  (busy_RAs < RA + RA_size) &
                                  (busy_Decs < Dec + Dec_size) &
                                  (busy_Decs > Dec - Dec_size))[0]) == 0

        condE = len(np.where((busy_RAs > RA + RA_size) &
                             (busy_RAs < RA + 3*RA_size) &
                             (busy_Decs < Dec + Dec_size) &
                             (busy_Decs > Dec - Dec_size))[0]) > 0

        condW = len(np.where((busy_RAs > RA - 3*RA_size) &
                             (busy_RAs < RA - RA_size) &
                             (busy_Decs < Dec + Dec_size) &
                             (busy_Decs > Dec - Dec_size))[0]) > 0

        condN = len(np.where((busy_RAs > RA - RA_size) &
                             (busy_RAs < RA + RA_size) &
                             (busy_Decs < Dec + 3*Dec_size) &
                             (busy_Decs > Dec + Dec_size))[0]) > 0

        condS = len(np.where((busy_RAs > RA - RA_size) &
                             (busy_RAs < RA + RA_size) &
                             (busy_Decs < Dec - Dec_size) &
                             (busy_Decs > Dec - 3*Dec_size))[0]) > 0

        if cond_empty & condN & condS & condE & condW:
            new_RAs.append(RA)
            new_Decs.append(Dec)
            busy_RAs = np.append(busy_RAs, RA)
            busy_Decs = np.append(busy_Decs, Dec)
            n += 1
            count = 0
        else:
            count += 1
        if count > 1000:
            print(f'found {n} empty spots over the {n_gals} requested')
            return new_RAs, new_Decs
    return new_RAs, new_Decs


def param_offset(param, fraction=0.02):
    offset = np.random.uniform(-fraction, fraction)
    param += param * offset
    return param


def sim_gal_and_clump(field,
                      seg_map,
                      sub_cat,
                      index,
                      gal_x, gal_y,
                      psf,
                      clump_cat,
                      gaussian,
                      px_scale=0.031,
                      flux_interval=[0.06, 0.3],
                      flux_tot_max=0.45,
                      pos_interval=[1, 2],
                      stamp_size=300):

    ''' Get galaxy parameters'''

    gal_n = param_offset(sub_cat[index]['sersic_index'])
    if gal_n > 6:
        gal_n = 6
    gal_hlr = param_offset(sub_cat[index]['radius'])
    mag = param_offset(sub_cat[index]['nircam_f444w_clear_magnitude'])
    gal_f = param_offset(mag2flux(mag))
    gal_q = 1 - sub_cat[index]['ellipticity']
    gal_q = param_offset(gal_q)

    ''' initialise stamps'''
    stamp_galaxy = galsim.ImageF(stamp_size, stamp_size, scale=px_scale)
    stamp_g_c = galsim.ImageF(stamp_size*3, stamp_size*3, scale=px_scale)
    # seg_g = galsim.ImageF(stamp_size, stamp_size, scale=px_scale)
    seg_g_c = galsim.ImageF(stamp_size*3, stamp_size*3, scale=px_scale)

    ''' sim galaxy'''
    galaxy = galsim.Sersic(gal_n, gal_hlr)
    galaxy = galaxy.shear(q=gal_q, beta=0*galsim.degrees)
    theta = np.random.uniform(0, 1) * 2 * np.pi * galsim.radians
    galaxy = galaxy.rotate(theta)
    # galaxy = galsim.Convolve(galaxy, psf)
    galaxy = galaxy.withFlux(gal_f)
    galaxy.drawImage(stamp_galaxy, method='no_pixel')

    "add gaussian to galaxy seg stamp"
    # seg_g.array = gaussian

    ''' nb of clumps and stamp init (with gal)'''
    if np.log10(gal_hlr < 0.8):
        n_clumps_max = 2
    else:
        n_clumps_max = 4
    n_clumps = np.random.randint(1, n_clumps_max+1)

    galaxy.drawImage(stamp_g_c, method='no_pixel')

    tot_flux = gal_f
    clumps_flux = 0
    companions = 1
    for _ in range(n_clumps):

        ''' sim clumps '''
        clump, clump_flux = sim_clump(gal_f, clumps_flux,
                                      stamp_size, px_scale,
                                      flux_interval,
                                      flux_tot_max,
                                      psf)

        if clump_flux > 0:
            # print(f'simulated {companions} clumps')
            tot_flux += clump_flux
            clumps_flux += clump_flux

            ''' get offset'''
            clump_x, clump_y = assign_clump_offset(gal_hlr, pos_interval,
                                                   px_scale)

            ''' Update clump catalogue'''
            clump_cat['gal_ID'].append(sub_cat['index'][index])
            clump_cat['flux'].append(clump_flux)
            clump_cat['percentage_flux'].append(clump_flux/gal_f)
            clump_cat['tot_percentage_flux'].append(tot_flux/gal_f)
            clump_cat['gal_x'].append(gal_x)
            clump_cat['gal_y'].append(gal_y)
            clump_cat['x_off'].append(clump_x)
            clump_cat['y_off'].append(clump_y)
            clump_cat['n_clumps'].append(1)
            for c in range(companions):
                clump_cat['n_clumps'][-1-c] = companions
                clump_cat['tot_percentage_flux'][-1-c] = tot_flux/gal_f
            companions += 1

            ''' Add galaxy and clump'''
            x_start, y_start = stamp_size+clump_x, stamp_size+clump_y   # stamp_size * 3 // 2 + clump_x - stamp_size // 2
            x_end, y_end = 2*stamp_size+clump_x, 2*stamp_size+clump_y

            # print(clump.array.shape, stamp_g_c.array.shape)
            # print(x_end-x_start, x_start, x_end, y_start, y_end)
            stamp_g_c.array[x_start:x_end, y_start:y_end] += clump.array
            seg_g_c.array[x_start:x_end, y_start:y_end] += gaussian

    ''' Add the clumpy galaxy to the field'''

    clumpy_galaxy_object = galsim.Image(np.ascontiguousarray(stamp_g_c.array.astype(np.float64)),
                                        scale=px_scale)

    clumpy_galaxy_object = galsim.InterpolatedImage(clumpy_galaxy_object)
    clumpy_galaxy_object = galsim.Convolve(clumpy_galaxy_object, psf)
    clumpy_galaxy_object = clumpy_galaxy_object.withFlux(gal_f)

    stamp_g_c = galsim.ImageF(stamp_size*3, stamp_size*3, scale=px_scale)
    clumpy_galaxy_object.drawImage(stamp_g_c, method='no_pixel')
    x_start, y_start = gal_x - stamp_size * 3 // 2, gal_y - stamp_size * 3 // 2
    x_end, y_end = gal_x + stamp_size * 3 // 2, gal_y + stamp_size * 3 // 2
    field[x_start:x_end, y_start:y_end] += stamp_g_c.array
    seg_map[x_start:x_end, y_start:y_end, 0] += seg_g_c.array
    seg_map[gal_x - stamp_size // 2:gal_x + stamp_size // 2,
            gal_y - stamp_size // 2:gal_y + stamp_size // 2, 1] += gaussian

    return (field, seg_map, stamp_galaxy.array, stamp_g_c.array, clump_cat)


def add_clumps_to_field(original_field, clumpy_field, cat,
                        busy_Ras, busy_Decs,
                        wcs, psf, n_gals,
                        mag_max, mag_min, hlr_min,
                        sigma_segmap=8,
                        px_scale=0.031,
                        show=False):

    seg_map = np.zeros((original_field.shape[0], original_field.shape[1], 2))
    RAs, Decs = find_empty_coords(busy_Ras, busy_Decs, n_gals, wcs)
    n_gals = len(RAs)
    sub_cat = create_sub_cat(cat, mag_max, mag_min, hlr_min)

    # if len(sub_cat) < len(RAs):
        # sub_cat = vstack(sub_cat, sub_cat)
    border_galaxies = 0
    clump_cat = {'ID': [],
                 'gal_ID': [],
                 'gal_x': [],
                 'gal_y': [],
                 'flux': [],
                 'x_off': [],
                 'y_off': [],
                 'n_clumps': [],
                 'percentage_flux': [],
                 'tot_percentage_flux': []
                 }

    first_clump_id = 0
    stamp_size = 300
    gaussian = generate_2d_gaussian(stamp_size//2, sigma_segmap)
    for i in range(n_gals):
        if i % 500 == 0:
            print(f'went through {i} galaxies')
        index = np.random.randint(1, len(sub_cat))
        gal_y, gal_x = wcs.wcs_world2pix(RAs[i], Decs[i], 1)
        gal_x = int(gal_x)
        gal_y = int(gal_y)
        try:
            (cw_clumps, seg_map,
                stamp_galaxy, stamp_g_c,
                clump_cat) = sim_gal_and_clump(clumpy_field,
                                               seg_map,
                                               sub_cat,
                                               index,
                                               gal_x, gal_y,
                                               psf,
                                               clump_cat,
                                               gaussian,
                                               stamp_size=stamp_size)

            if (i < show):
                gal_hlr = sub_cat[index]['radius']
                gal_f = sub_cat[index]['nircam_f444w_clear_magnitude']
                n_clumps = clump_cat['n_clumps'][first_clump_id]
                fig = inspect_clumps_creation(original_field,
                                                clumpy_field,
                                                stamp_galaxy,
                                                stamp_g_c,
                                                gal_x, gal_y,
                                                gal_hlr, n_clumps,
                                                clump_cat,
                                                first_clump_id,
                                                gal_f,
                                                px_scale,
                                                seg_map)

                fig.savefig(f'plots/inspect_clumps_simu_{i}')
                first_clump_id += n_clumps

        except ValueError:
            print("pb in gal ",  i, gal_x, gal_y)
            for _ in range(clump_cat['n_clumps'][-1]):
                clump_cat['gal_ID'].pop()
                clump_cat['flux'].pop()
                clump_cat['gal_x'].pop()
                clump_cat['gal_y'].pop()
                clump_cat['x_off'].pop()
                clump_cat['y_off'].pop()
                clump_cat['n_clumps'].pop()
                border_galaxies += 1

    fraction_simulated = 1 - border_galaxies/n_gals
    print(f'simulated {n_gals - border_galaxies} instead of {n_gals},')
    print(f' i.e {fraction_simulated}')

    ''' add the gaussian to the galaxy without clumps'''
    for i in range(len(cat)):
        if cat['nircam_f444w_clear_magnitude'][i] > 26:
            continue
        gal_y, gal_x = wcs.wcs_world2pix(busy_Ras[i], busy_Decs[i], 1)
        gal_x = int(gal_x)
        gal_y = int(gal_y)
        try:
            seg_map[gal_x-stamp_size//2:gal_x+stamp_size//2,
                    gal_y-stamp_size//2:gal_y+stamp_size//2, 1] += gaussian
        except ValueError:
            continue
    return (cw_clumps, clump_cat, seg_map)


def create_x_y(mosaic, cat, stamp_size,
               x_start=0, y_start=0,
               border=3,
               clumps_only=None,
               show=False,
               n_stamps=None):

    fig, ax = plt.subplots(10, 2, figsize=(10, 30))
    i = 0
    plotted = 0
    sx, sy = mosaic.shape
    if not n_stamps:
        n_stamps = (sx-x_start)//stamp_size * (sy-y_start)//stamp_size
    for ix in range((sx-x_start)//stamp_size):
        for iy in range((sy-y_start)//stamp_size):
            i += 1
            if i > n_stamps:
                break
            xs, xe = x_start+ix*stamp_size, x_start+ix*stamp_size+stamp_size
            ys, ye = y_start+iy*stamp_size, y_start+iy*stamp_size+stamp_size
            stamp = mosaic[xs:xe, ys:ye]

            clumps_idx = np.where((cat['gal_x'] > xs) &
                                  (cat['gal_x'] < xe) &
                                  (cat['gal_y'] > ys) &
                                  (cat['gal_y'] < ye))[0]

            hdu = fits.PrimaryHDU(stamp)
            hdulist = fits.HDUList([hdu])

            hdulist.writeto(f'./training_images/train_img_{i}.fits',
                            overwrite=True)
            label_i = f'../data/training_images/train_img_{i}.fits'
            if len(clumps_idx) > 0:
                for c in clumps_idx:
                    cond1 = (cat['gal_x'][c] + cat['x_off'][c]) > xe - border
                    cond2 = (cat['gal_x'][c] + cat['x_off'][c]) < xs + border
                    cond3 = (cat['gal_y'][c] + cat['y_off'][c]) < ys + border
                    cond4 = (cat['gal_y'][c] + cat['y_off'][c]) > ye - border
                    if cond1 | cond2 | cond3 | cond4:
                        continue
                    x_clump = cat['gal_x'][c]-xs+cat['x_off'][c]
                    y_clump = cat['gal_y'][c]-ys+cat['y_off'][c]
                    label_i = f'{label_i} 5,5,{int(x_clump)},{int(y_clump)},1 '
                    label_i = f"{label_i} 5,5,{int(cat['gal_x'][c]-xs)},{int(cat['gal_y'][c]-ys)},2 "
                    if (plotted < show):
                        interval = ZScaleInterval()
                        a = interval.get_limits(stamp)
                        ax[plotted, 0].imshow(stamp, vmin=a[0], vmax=a[1],
                                              cmap='bone', origin='lower')
                        ax[plotted, 1].imshow(clumps_only[xs:xe, ys:ye],
                                              cmap='bone', origin='lower')

                        ax[plotted, 1].scatter(cat['gal_y'][c]-ys,
                                               cat['gal_x'][c]-xs,
                                               c='blue', marker='o')

                        plot_box_from_start(ax[plotted, 1],
                                            x_clump,
                                            y_clump,
                                            s_star=5)

                with open("training_labels_with_gals.txt", "a") as text_file:
                    text_file.write(label_i + "\n")
                    plotted += 1

    return fig


def generate_2d_gaussian(stamp_size, sigma):

    # Initializing the random seed
    random_seed = 1000

    # List containing the variance
    # covariance values
    cov_val = 0

    # Setting mean of the distributino
    # to be at (0,0)
    mean = np.array([0, 0])

    # Initializing the covariance matrix
    cov = np.array([[sigma, cov_val], [cov_val, sigma]])

    # Generating a Gaussian bivariate distribution
    # with given mean and covariance matrix
    distr = multivariate_normal(cov=cov, mean=mean,
                                seed=random_seed)

    # Generating a meshgrid
    x = np.arange(-stamp_size, stamp_size)
    y = np.arange(-stamp_size, stamp_size)
    X, Y = np.meshgrid(x, y)

    # Generating the density function
    # for each point in the meshgrid
    pdf = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pdf[i, j] = distr.pdf([X[i, j], Y[i, j]])
    pdf = (pdf-np.min(pdf)) / (np.max(pdf)-np.min(pdf))
    return pdf

# psf_scipy = fits.open('raw/NIRCam_f444w.fits')[0].data
    # gc_array = stamp_g_c.array
    # print(gc_array)
    # print(np.shape(stamp_g_c.array), np.shape(psf_scipy))
    # convolved_clumpy_galaxy = scipy_convolve(gc_array, psf_scipy, mode='same', method='fft')
    # print(np.shape(convolved_clumpy_galaxy))
    # stamp_g_c.array = convolved_clumpy_galaxy

    # hdu = fits.PrimaryHDU(seg_g_c.array)
    # hdulist = fits.HDUList([hdu])
    # temp_fits_file = './temp_fits.fits'
    # hdulist.writeto(temp_fits_file,
    #                 overwrite=True)
    # clumpy_galaxy_object = galsim.Image(np.ascontiguousarray(stamp_g_c.array.astype(np.float64)),
                                        # scale=px_scale)

    # clumpy_galaxy_object = galsim.InterpolatedImage(clumpy_galaxy_object)
    # clumpy_galaxy_object = galsim.InterpolatedImage(temp_fits_file)
    # clumpy_galaxy_object = galsim.Image(stamp_g_c.array, scale=px_scale)
    # clumpy_galaxy_object = clumpy_galaxy_object.withFlux(gal_f)
    # clumpy_galaxy_object = galsim.Convolve(clumpy_galaxy_object, psf)
    # stamp_g_c = galsim.ImageF(stamp_size*3, stamp_size*3, scale=px_scale)
    # clumpy_galaxy_object.drawImage(stamp_g_c, method='no_pixel')
