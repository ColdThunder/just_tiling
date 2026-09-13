import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import healpy as hp

# better to use hyperuniform_plotstyle.mplstyle
#plt.style.use("./hyperuniform_plotstyle.mplstyle")


def plot_gal_den_map(
    counts_map_plot,
    minv = 0,
    maxv = None,
    rot = [30,30,0],
    half_sky = True,
    badcolor = (0,0,0,0.1),
    savefile = "./testplot/galaxy_counts_map.pdf",
    showfig = True,
    ):

    with plt.style.context("./hyperuniform_plotstyle.mplstyle"):
        hp.orthview(
            counts_map_plot, 
            title = "",
            unit = "galaxy counts",
            cmap = "rainbow",
            nest = False,
            min = minv,
            max = maxv,
            rot = rot,
            half_sky = half_sky,
            badcolor = badcolor
            )

        hp.graticule()

        if savefile is not None:
            plt.savefig(savefile,bbox_inches='tight')
        if showfig:
            plt.show()

def plot_tile_distribution(
    background_map,
    ra,
    dec,
    minv = None,
    maxv = None,
    rot = [30,30,0],
    cmap = "Oranges", 
    bkgcolor = "orange",
    bkgtext = "clusters",
    half_sky = True,
    badcolor = (0,0,0,0.1),
    savefile = "./testplot/tile_distribution_with_background.pdf",
    showfig = True,
    tile_unit_deg = True,
    ):

    if tile_unit_deg:
        phi = np.deg2rad(ra)
        theta = np.deg2rad(90 - dec)
    else:
        phi = ra
        theta = 90 -dec

    from matplotlib.patches import Rectangle, Circle
    with plt.style.context("./hyperuniform_plotstyle.mplstyle"):
        hp.orthview(
            background_map, 
            title = "",
            unit = "",
            cmap = cmap,
            min = minv,
            max = maxv,
            nest=False,
            rot = rot,
            half_sky = True,
            badcolor = badcolor,
            cbar=False,
            )

        hp.projscatter(theta,phi, s=1, c='black',marker='.',fc='black',ec=None,lw=0,alpha=1)

        hp.graticule()

        ax = plt.gca()

        items = [
            ("square", bkgcolor, bkgtext ),
            ("circle", "black",  "tiles"),
            ]

        x = 0.72
        y_start = 0.90
        dy = 0.055

        w, h = 0.045, 0.032
        point_radius = 0.003

        ax.add_patch(Rectangle(
            (x - 0.025, y_start - dy - 0.020),
            0.25,
            2 * dy + 0.025,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.85,
            clip_on=False,
            zorder=90,
            ))

        for i, (shape, color, label) in enumerate(items):
            y = y_start - i * dy
            center_y = y + h / 2

            if shape == "square":
                 symbol = Rectangle(
                    (x, y),
                    w,
                    h,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor="none",
                    linewidth=0.8,
                    alpha=0.6,
                    clip_on=False,
                    zorder=100,
                 )
            else:
                symbol = Circle(
                    (x + w / 2, center_y),
                    radius=point_radius,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor="none",
                    clip_on=False,
                    zorder=100,
                )

            ax.add_patch(symbol)

            ax.text(
                x + w + 0.015,
                center_y,
                label,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=10,
                color="black",
                clip_on=False,
                zorder=100,
                )

        if savefile is not None:
            plt.savefig(savefile, bbox_inches='tight')
        if showfig:
            plt.show()

def plot_color_tile_distribution(
    mask_map,
    ra,
    dec,
    colors,
    colorcode = "rainbow",
    rot = [30,30,0],
    half_sky = True,
    badcolor = (0,0,0,0.1),
    savefile = "./testplot/tile_distribution_with_colors.pdf",
    showfig = True,
    tile_unit_deg = True,
    ):

    if tile_unit_deg:
        phi = np.deg2rad(ra)
        theta = np.deg2rad(90 - dec)
    else:
        phi = ra
        theta = 90 -dec

    with plt.style.context("./hyperuniform_plotstyle.mplstyle"):
        hp.orthview(
            mask_map,
            title = "",
            nest = False,
            rot = rot,
            half_sky = half_sky,
            cmap='Greys',
            min=0,max=100,
            badcolor = badcolor,
            cbar=False
        )

        hp.projscatter(theta,phi,c=colors,cmap=colorcode, s=0.01, marker='*')

        hp.graticule()

        if savefile is not None:
            plt.savefig(savefile, bbox_inches='tight')
        if showfig:
            plt.show()



def draw_spherical_circles_advance(
    lon_centers,
    lat_centers,
    radius_deg,
    color="black",
    linewidth=0.3,
    alpha=1.0,
    npoints=16,
    rasterized=False,
):

    from matplotlib.collections import LineCollection

    """
    plot a lot of circles

    lon_centers, lat_centers in rad
    radius_deg
    """

    lon0 = np.asarray(lon_centers, dtype=float)[:, None]
    lat0 = np.asarray(lat_centers, dtype=float)[:, None]

    radius = np.radians(radius_deg)

    bearing = np.linspace(
        0,
        2 * np.pi,
        npoints,
        endpoint=True,
    )[None, :]

    lat = np.arcsin(
        np.sin(lat0) * np.cos(radius)
        + np.cos(lat0) * np.sin(radius) * np.cos(bearing)
    )

    lon = lon0 + np.arctan2(
        np.sin(bearing) * np.sin(radius) * np.cos(lat0),
        np.cos(radius) - np.sin(lat0) * np.sin(lat),
    )

    ax = plt.gca()

    circle_shape = lon.shape

    x, y = ax.proj.ang2xy(
        np.degrees(lon).ravel(),
        np.degrees(lat).ravel(),
        lonlat=True,
    )

    x = np.ma.asarray(x).filled(np.nan).reshape(circle_shape)
    y = np.ma.asarray(y).filled(np.nan).reshape(circle_shape)

    if np.ma.isMaskedArray(x):
        x = x.filled(np.nan)

    if np.ma.isMaskedArray(y):
        y = y.filled(np.nan)

    x = np.asarray(x)
    y = np.asarray(y)

    segments = np.stack([x, y], axis=-1)

    collection = LineCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
        rasterized=rasterized,
        zorder=20,
    )

    ax.add_collection(collection)

    return collection




def plot_tile_circles(
    background_map,
    ra,
    dec,
    minv = None,
    maxv = None,
    rot = [30,30,0],
    cmap = "Oranges", 
    bkgcolor = "orange",
    bkgtext = "clusters",
    half_sky = True,
    badcolor = (0,0,0,0.1),
    savefile = "./testplot/tile_circles_with_background.pdf",
    showfig = True,
    tile_unit_deg = True,
    ):

    radius = 0.5968 # deg
 
    if tile_unit_deg:
        phi = np.deg2rad(ra)
        theta = np.deg2rad(90 - dec)
    else:
        phi = ra
        theta = 90 -dec

    from matplotlib.patches import Rectangle, Circle
    with plt.style.context("./hyperuniform_plotstyle.mplstyle"):
        hp.orthview(
            background_map,
            title = "",
            unit = "",
            cmap = cmap,
            min = minv,
            max = maxv,
            nest=False,
            rot = rot,
            half_sky = True,
            badcolor = badcolor,
            cbar=False,
            )

        draw_spherical_circles_advance(
            phi,
            np.pi/2-theta,
            radius,
            color="black",
            linewidth=0.1,
            alpha=1,
            )
         
        hp.graticule()

        ax = plt.gca()

        items = [
            ("square", bkgcolor , bkgtext),
            ("circle", "black",  "tiles"),
            ]
        nitem=len(items)

        x = 0.72
        y_start = 0.90
        dy = 0.055

        w, h = 0.045, 0.032
        point_radius = 0.016

        ax.add_patch(Rectangle(
            (x - 0.025, y_start - dy - 0.020),
            0.25,
            nitem * dy + 0.025,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.85,
            clip_on=False,
            zorder=90,
            ))

        for i, (shape, color, label) in enumerate(items):
            y = y_start - i * dy
            center_y = y + h / 2

            if shape == "square":
                symbol = Rectangle(
                    (x, y),
                    w,
                    h,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.6,
                    clip_on=False,
                    zorder=100,
                )
            else:
                symbol = Circle(
                    (x + w / 2, center_y),
                    radius=point_radius,
                    transform=ax.transAxes,
                    facecolor="white",
                    edgecolor=color,
                    clip_on=False,
                    zorder=100,
                    )

            ax.add_patch(symbol)

            ax.text(
                x + w + 0.015,
                center_y,
                label,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=10,
                color="black",
                clip_on=False,
                zorder=100,
                )

        if savefile is not None:
            plt.savefig(savefile,bbox_inches='tight')
        if showfig:
            plt.show()


