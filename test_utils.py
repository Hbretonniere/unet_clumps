from astropy.io import fits
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import numpy as np


def find_closest_point(coords, cat):
    '''
        coords :  the clump (x, y) coordinates we want
                  to match to our catalogue
        cat : the catalogue to search in
        return : the catalog's index of the point closest
                 to the clicked coordinate

    '''
    # Defining the table we want to match for
    searching_for = np.zeros((1, 2))
    searching_for[0, 0] = coords[0]
    searching_for[0, 1] = coords[1]

    # Defining the table we want to match with
    searching_in = np.zeros((len(cat), 2))
    searching_in[:, 0] = cat[:, 0]
    searching_in[:, 1] = cat[:, 1]

    # find the closest neighboor
    _, match_index = KDTree(searching_in).query(searching_for)

    closestx, closesty = searching_in[match_index][0]

    return match_index[0], closestx, closesty


def find_clumps(img, window_size):
    coordinates = peak_local_max(img, min_distance=window_size)
    return coordinates


def compute_class(clump_coords, clump_cat_x_y, galaxy_cat_x_y, margin):

    'find closest true clump'
    _, closest_x_c, closest_y_c = find_closest_point(clump_coords,
                                                     clump_cat_x_y)
    dist_c = np.sqrt((clump_coords[0]-closest_x_c)**2 +
                     (clump_coords[1]-closest_y_c)**2)

    'find closest true galaxy'
    _, closest_x_g, closest_y_g = find_closest_point(clump_coords,
                                                     galaxy_cat_x_y)
    dist_g = np.sqrt((clump_coords[0]-closest_x_c)**2 +
                     (clump_coords[1]-closest_y_c)**2)

    if dist_g < dist_c:
        return 'FPg'
    
    if dist_c < margin:
        return 'TP'

    if dist_c > margin:
        return 'FP'

def compute_completeness_purity(pred_cat, galaxy_cat, clump_cat):
    galaxy_cat_x_y = np.zeros((len(galaxy_cat), 2))
    galaxy_cat_x_y[:, 0] = galaxy_cat_x_y['X']
    galaxy_cat_x_y[:, 1] = galaxy_cat_x_y['Y']
    for i in range(len(pred_cat)):
        
