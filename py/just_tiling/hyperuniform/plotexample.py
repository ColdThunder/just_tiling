
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import healpy as hp


# plot galaxy count map
if False:
    from hyperuniform_plots import plot_gal_den_map

    counts_map_plot = np.load('./testdata/gal_counts_map.npz')
    plot_gal_den_map(
        counts_map_plot,
        savefile = "./testplot/galaxy_counts_map.pdf",
        )



# plot tile distribution
if True:
    from hyperuniform_plots import plot_tile_distribution

    # make a zero-valued trivial map with mask
    data = np.load('./testdata/mask_from_gal.npz')
    mask = data[data.files[0]]
    npix = len(mask)
    trivial_map = np.zeros(npix)
    trivial_map[mask] = hp.UNSEEN
    trivial_map[~mask] = 0.

    tile_coord = np.load('./testdata/hyper_final_tile.npz')
    ra = tile_coord["tile_coord"][:,0]
    dec = tile_coord["tile_coord"][:,1]

    plot_tile_distribution(
        trivial_map, ra, dec, 
        minv=0, maxv=1, bkgtext="inside",
        savefile = "./testplot/tile_distribution_with_background.pdf",
        )
    #use the most left edge of cmap = "Oranges"



# plot tile distribution with cluster region
if False:
    from hyperuniform_plots import plot_tile_distribution

    # read in cluster region
    data = np.load('./testdata/cluster_region_map.npz')
    cluster_region = data[data.files[0]]

    tile_coord = np.load('./testdata/hyper_final_tile.npz')
    ra = tile_coord["tile_coord"][:,0]
    dec = tile_coord["tile_coord"][:,1]

    plot_tile_distribution(
        cluster_region, ra, dec, 
        minv=0, maxv=10,
        savefile="./testplot/tile_with_cluster_region.pdf",
        )
    # valid region = 0, cluster region = 2, set edges of cmap as 0, 10



# plot tile distribution with galaxy count map
if False:
    from hyperuniform_plots import plot_tile_distribution

    counts_map_plot = np.load('./testdata/gal_counts_map.npz')

    tile_coord = np.load('./testdata/hyper_final_tile.npz')
    ra = tile_coord["tile_coord"][:,0]
    dec = tile_coord["tile_coord"][:,1]

    plot_tile_distribution(
        counts_map_plot, ra, dec,
        cmap="rainbow",
        bkgcolor = "orange",
        bkgtext = "gal_den",
        savefile = "./testplot/tile_with_gal_den.pdf",
        )


# plot tile distribution with color
if False:
    from hyperuniform_plots import plot_color_tile_distribution
    
    data = np.load('./testdata/mask_from_gal.npz')
    mask = data[data.files[0]]
    npix = len(mask)
    trivial_map = np.zeros(npix)
    trivial_map[mask] = hp.UNSEEN
    trivial_map[~mask] = 0.

    tile_coord = np.load('./testdata/hyper_final_tile.npz')
    ra = tile_coord["tile_coord"][:,0]
    dec = tile_coord["tile_coord"][:,1]
    data = np.load('./testdata/hyper_tile_colors.npz')
    colors = data[data.files[0]]

    plot_color_tile_distribution(
        trivial_map, ra, dec, colors,
        savefile = "./testplot/tile_distribution_with_colors.pdf",
        )


# plot tile circles with cluster region
if True:
    from hyperuniform_plots import plot_tile_circles

    # read in cluster region
    data = np.load('./testdata/cluster_region_map.npz')
    cluster_region = data[data.files[0]]

    tile_coord = np.load('./testdata/hyper_final_tile.npz')
    ra = tile_coord["tile_coord"][:,0]
    dec = tile_coord["tile_coord"][:,1]

    plot_tile_circles(
        cluster_region, ra, dec,
        minv=0, maxv=10,
        savefile = "./testplot/tile_circles_with_background.pdf",
        )
    # valid region = 0, cluster region = 2, set edges of cmap as 0, 10
   


