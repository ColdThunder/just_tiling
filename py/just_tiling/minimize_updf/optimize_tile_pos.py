import os
import time, sys
import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from scipy.optimize import minimize, Bounds
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def _ra_minmax_deg_diff(ra_corners):
    """Difference between Max-Min of RA [deg] for corners that may straddle the 0/360 boundary.

    Parameters
    ----------
    ra_corners : :class:`array_like`
        RA values of pixel corners in degrees (any wrapping around 0/360).

    Returns
    -------
    :class:`float`
        Angular RA span in degrees, accounting for the 0/360 wrap.
    """
    ra = np.asarray(ra_corners, dtype=np.float64) % 360.0
    if np.ptp(ra) <= 180.0:
        ra_diff = ra.max() - ra.min()
    else:
        ra_diff = ra.min() + 360.0 - ra.max()
    return ra_diff


def get_pixel_radec_diff_corners(NSIDE, ipix, nest=False):
    """Return RA width and Dec height (deg) of a HEALPix pixel from its 4 corners.

    Parameters
    ----------
    NSIDE : :class:`int`
        HEALPix ``nside`` of the pixelization.
    ipix : :class:`int`
        HEALPix pixel index.
    nest : :class:`bool`, optional, defaults to ``False``
        If ``True``, use NESTED ordering; otherwise RING.

    Returns
    -------
    ra_diff : :class:`float`
        Angular width in RA (degrees), accounting for the 0/360 wrap.
    dec_diff : :class:`float`
        Angular height in Dec (degrees).
    """
    # hp.boundaries returns shape (3, 4): each column is xyz vector of one corner
    corners_xyz = hp.boundaries(NSIDE, ipix, nest=nest)
    ra_corners = np.empty(4, dtype=np.float64)
    dec_corners = np.empty(4, dtype=np.float64)

    for i in range(4):
        # vec2ang returns 1-element arrays when input is single vector
        ra_arr, dec_arr = hp.vec2ang(corners_xyz[:, i], lonlat=True)
        # extract scalar float to suppress deprecation warning
        ra_corners[i] = ra_arr.item()
        dec_corners[i] = dec_arr.item()

    ra_diff = _ra_minmax_deg_diff(ra_corners)
    dec_diff = dec_corners.max() - dec_corners.min()
    return ra_diff, dec_diff


def pixel_center_to_edge_deg(nside, pixid, nest=False):
    """Maximum angular distance [deg] from pixel center to a corner (pixel edge).

    Parameters
    ----------
    nside : :class:`int`
        HEALPix ``nside`` of the pixelization.
    pixid : :class:`int`
        HEALPix pixel index.
    nest : :class:`bool`, optional, defaults to ``False``
        If ``True``, use NESTED ordering; otherwise RING.

    Returns
    -------
    :class:`float`
        Maximum center-to-corner angular separation in degrees.
    """
    center = hp.pix2vec(nside, pixid, nest=nest)
    corners = hp.boundaries(nside, pixid, nest=nest)
    seps = []
    for i in range(4):
        dot = np.clip(np.dot(center, corners[:, i]), -1.0, 1.0)
        seps.append(np.degrees(np.arccos(dot)))
    return float(max(seps))


def select_gal_idx_ext_near_pixel(gal_cat, neigh_ipix, pixid, nside, nest=False):
    """Select galaxies in neighboring pixels near the target pixel edge.

    Galaxies are kept if they lie within half a pixel size beyond the
    target pixel edge. Uses HEALPix resolution and center-to-edge distance
    instead of corner RA/Dec spans, which overestimate pixel size
    especially in RA.

    Parameters
    ----------
    gal_cat : :class:`~astropy.table.Table`
        Galaxy catalog with columns ``ra``, ``dec``, and ``pixid``.
    neigh_ipix : :class:`array_like`
        HEALPix indices of neighboring pixels.
    pixid : :class:`int`
        Target HEALPix pixel index.
    nside : :class:`int`
        HEALPix ``nside`` of the pixelization.
    nest : :class:`bool`, optional, defaults to ``False``
        If ``True``, use NESTED ordering; otherwise RING.

    Returns
    -------
    :class:`~numpy.ndarray`
        Integer indices into ``gal_cat`` for selected neighbor galaxies.
        Empty array if there are no neighbors or no candidates.

    Notes
    -----
        - No lower bound on separation: neighbor pixels can contain
          sources that are geometrically inside the edge near HEALPix
          vertices.
    """
    if neigh_ipix.size == 0:
        return np.array([], dtype=np.int64)

    pix_ra, pix_dec = hp.pix2ang(nside, pixid, nest=nest, lonlat=True)
    pixel_size = hp.nside2resol(nside, arcmin=True) / 60.0
    half_pix_size = 0.5 * pixel_size
    pix_edge = pixel_center_to_edge_deg(nside, pixid, nest=nest)

    gal_idx_cand = np.where(np.isin(gal_cat["pixid"], neigh_ipix))[0]
    if gal_idx_cand.size == 0:
        return gal_idx_cand

    ra = np.asarray(gal_cat["RA"][gal_idx_cand], dtype=np.float64)
    dec = np.asarray(gal_cat["DEC"][gal_idx_cand], dtype=np.float64)
    center = SkyCoord(pix_ra * u.deg, pix_dec * u.deg)
    sep = center.separation(SkyCoord(ra * u.deg, dec * u.deg)).to_value(u.deg)

    # Neighbor galaxies within half a pixel beyond the target pixel edge.
    # No lower bound: neighbor pixels can contain sources geometrically
    # inside the edge near HEALPix vertices.
    near = sep <= pix_edge + half_pix_size
    return gal_idx_cand[near]


def cal_unobs_tile_window(r, alpha, R1, A, unobs_const):
    """Unobservable probability window as a function of angular radius.

    Parameters
    ----------
    r : :class:`array_like` or :class:`float`
        Angular separation between target and tile center (degrees),
        mapped into the radial window argument.
    alpha : :class:`float`
        Power-law slope of the window.
    R1 : :class:`float`
        Outer radius of the tile in degrees.
    A : :class:`float`
        Amplitude constant of the window (often set so pdf=1 at
        ``r_ext_factor * R1``).
    unobs_const : :class:`float`
        Baseline unobservable probability inside the tile annulus.

    Returns
    -------
    :class:`array_like` or :class:`float`
        Unobservable probability from
        ``A * ((r / R1)**alpha - 1) + unobs_const``.
    """
    r_norm = r/R1
    return A*(r_norm**alpha -1.0) + unobs_const

def _unobs_for_one_tile(tile_ra, tile_dec, gal_ra, gal_dec, gal_unobs, R0, R1, alpha, A, unobs_const, r_ext_factor=2.5):
    """Per-galaxy unobservable PDF for a single tile (parallel worker).

    Parameters
    ----------
    tile_ra : :class:`float`
        Tile center RA in degrees.
    tile_dec : :class:`float`
        Tile center Dec in degrees.
    gal_ra : :class:`array_like`
        Galaxy RAs in degrees.
    gal_dec : :class:`array_like`
        Galaxy Decs in degrees.
    gal_unobs : :class:`array_like`
        Per-galaxy unobservable weights (unused in the PDF shape, kept
        for a consistent worker signature).
    R0 : :class:`float`
        Inner radius of the tile in degrees.
    R1 : :class:`float`
        Outer radius of the tile in degrees.
    alpha : :class:`float`
        Power-law slope of the unobservable window.
    A : :class:`float`
        Amplitude constant of the unobservable window.
    unobs_const : :class:`float`
        Baseline unobservable probability for ``R0 <= r < R1``.
    r_ext_factor : :class:`float`, optional, defaults to 2.5
        Maximum radius in units of ``R1`` where the exterior window
        is applied.

    Returns
    -------
    :class:`~numpy.ndarray`
        Per-galaxy unobservable PDF values (length ``len(gal_ra)``).
        Default is 1 (uncovered / fully unobservable contribution).

    Notes
    -----
        - Interior (``0 < r < R0``): mirrored radial window about
          ``Rm = (R0 + R1) / 2``.
        - Annulus (``R0 <= r < R1``): constant ``unobs_const``.
        - Exterior (``R1 <= r < r_ext_factor * R1``): rising window.
    """
    Rm = (R0 + R1) / 2.0
    tile_coord = SkyCoord(tile_ra, tile_dec, frame='icrs', unit='deg')
    gal_coord = SkyCoord(gal_ra, gal_dec, frame='icrs', unit='deg')
    sep = tile_coord.separation(gal_coord).to_value(u.deg)

    # unobserved probability distributino function
    gal_unobs_pdf = np.ones(len(gal_ra), dtype=np.float64)

    # Index into full galaxy arrays using the tile mask
    mask0 = (sep > 0.0) & (sep < R0)
    r00 = -1 * (sep[mask0] - Rm)
    gal_unobs_pdf[mask0] = cal_unobs_tile_window(r00, alpha, R1, A, unobs_const)

    mask1 = (sep >= R1) & (sep < r_ext_factor * R1)
    r11 = sep[mask1]
    gal_unobs_pdf[mask1] = cal_unobs_tile_window(r11, alpha, R1, A, unobs_const)

    mask2 = (sep >= R0)&(sep < R1)
    gal_unobs_pdf[mask2] = unobs_const

    return gal_unobs_pdf


def cal_total_unobs_pdf(tiles_radec, gal_ra, gal_dec, gal_unobs, R0, R1, alpha, A, unobs_const, r_ext_factor=2.5, n_jobs=-1):
    """Total unobservable probability for a set of tiles and galaxies.

    For each galaxy, multiplies per-tile unobservable PDFs and weights
    by ``gal_unobs``. Uncovered galaxies contribute a factor of 1.

    Parameters
    ----------
    tiles_radec : :class:`array_like`
        Flattened or ``(n_tiles, 2)`` array of tile ``(RA, Dec)`` in
        degrees.
    gal_ra : :class:`array_like`
        Galaxy RAs in degrees.
    gal_dec : :class:`array_like`
        Galaxy Decs in degrees.
    gal_unobs : :class:`array_like`
        Per-galaxy unobservable weights.
    R0 : :class:`float`
        Inner radius of the tile in degrees.
    R1 : :class:`float`
        Outer radius of the tile in degrees.
    alpha : :class:`float`
        Power-law slope of the unobservable window.
    A : :class:`float`
        Amplitude constant of the unobservable window.
    unobs_const : :class:`float`
        Baseline unobservable probability inside the tile annulus.
    r_ext_factor : :class:`float`, optional, defaults to 2.5
        Maximum radius in units of ``R1`` for the exterior window.
    n_jobs : :class:`int`, optional, defaults to -1
        Number of parallel jobs for per-tile PDF evaluation
        (``joblib``; ``-1`` uses all CPUs).

    Returns
    -------
    :class:`float`
        Scalar total unobservable PDF
        ``sum_g [prod_tiles pdf(tile, g) * gal_unobs[g]]``.
    """
    tiles_radec = np.asarray(tiles_radec).reshape(-1, 2)
    tiles_ra, tiles_dec = tiles_radec[:, 0], tiles_radec[:, 1]
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_unobs_for_one_tile)(tile_ra, tile_dec, gal_ra, gal_dec, gal_unobs, R0, R1, alpha, A, unobs_const, r_ext_factor=r_ext_factor)
        for tile_ra, tile_dec in zip(tiles_ra, tiles_dec)
    )
    pdf_array = np.stack([res for res in results], axis=0)  # (n_tiles, n_gal)
    
    total_pdf = np.dot(np.prod(pdf_array, axis=0), gal_unobs)
    return total_pdf


def optimize_tile_pos_fun(tiles_ra, tiles_dec, ra_bound, dec_bound, gal_ra, gal_dec, gal_unobs, R0, R1, alpha, A, unobs_const, r_ext_factor, n_jobs):
    """Optimize tile (RA, Dec) positions to minimize total unobservable PDF.

    Parameters
    ----------
    tiles_ra : :class:`array_like`
        Initial tile center RAs in degrees.
    tiles_dec : :class:`array_like`
        Initial tile center Decs in degrees.
    ra_bound : :class:`float`
        Maximum allowed RA shift (degrees) from the initial position.
    dec_bound : :class:`float`
        Maximum allowed Dec shift (degrees) from the initial position.
    gal_ra : :class:`array_like`
        Galaxy RAs in degrees used in the objective.
    gal_dec : :class:`array_like`
        Galaxy Decs in degrees used in the objective.
    gal_unobs : :class:`array_like`
        Per-galaxy unobservable weights.
    R0 : :class:`float`
        Inner radius of the tile in degrees.
    R1 : :class:`float`
        Outer radius of the tile in degrees.
    alpha : :class:`float`
        Power-law slope of the unobservable window.
    A : :class:`float`
        Amplitude constant of the unobservable window.
    unobs_const : :class:`float`
        Baseline unobservable probability inside the tile annulus.
    r_ext_factor : :class:`float`
        Maximum radius in units of ``R1`` for the exterior window.
    n_jobs : :class:`int`
        Number of parallel jobs passed to :func:`cal_total_unobs_pdf`.

    Returns
    -------
    res : :class:`~scipy.optimize.OptimizeResult`
        Result of ``scipy.optimize.minimize`` with method
        ``trust-constr``. Optimized positions are in ``res.x``
        (flattened ``[RA, Dec, ...]``).

    Notes
    -----
        - Bounds are box constraints around the initial tile centers:
          ``[RA +/- ra_bound, Dec +/- dec_bound]``.
    """
    # set the bounds for tile (ra, dec) position optimization
    tiles_ra = np.asarray(tiles_ra, dtype=np.float64)
    tiles_dec = np.asarray(tiles_dec, dtype=np.float64)
    n_opt_tiles = len(tiles_ra)
    tiles_radec = np.column_stack([tiles_ra, tiles_dec])

    lb_flat = np.empty(2 * n_opt_tiles, dtype=np.float64)
    ub_flat = np.empty(2 * n_opt_tiles, dtype=np.float64)
    lb_flat[0::2] = tiles_ra - ra_bound
    lb_flat[1::2] = tiles_dec - dec_bound
    ub_flat[0::2] = tiles_ra + ra_bound
    ub_flat[1::2] = tiles_dec + dec_bound
    bounds = Bounds(lb_flat, ub_flat)

    res = minimize(
        cal_total_unobs_pdf,
        tiles_radec.flatten(),
        args=(gal_ra, gal_dec, gal_unobs, R0, R1, alpha, A, unobs_const, r_ext_factor, n_jobs),
        method="trust-constr",
        bounds=bounds,
        options={"maxiter": 20, "gtol": 10},
    )
    return res


def process_one_pixid(
    pixid,
    tiles,
    gal_cat,
    Nside,
    Npasses,
    unobs_3pass,
    R0,
    R1,
    alpha,
    A,
    unobs_const,
    r_ext_factor,
    n_jobs,
):
    """Run tile-position optimization for one HEALPix pixel.

    Selects tiles and galaxies in ``pixid``, adds near-edge galaxies
    from neighboring pixels, then minimizes the total unobservable PDF.

    Parameters
    ----------
    pixid : :class:`int`
        HEALPix pixel index to process.
    tiles : :class:`~astropy.table.Table`
        Tile table with columns ``RA``, ``DEC``, ``TILEID``, ``pixid``.
    gal_cat : :class:`~astropy.table.Table`
        Galaxy catalog with columns ``ra``, ``dec``, ``pixid``.
    Nside : :class:`int`
        HEALPix ``nside`` used for pixel assignment.
    Npasses : :class:`int`
        Number of survey passes (used to scale RA/Dec move bounds).
    unobs_3pass : :class:`float`
        Unobservable weight applied to external (neighbor) galaxies.
    R0 : :class:`float`
        Inner radius of the tile in degrees.
    R1 : :class:`float`
        Outer radius of the tile in degrees.
    alpha : :class:`float`
        Power-law slope of the unobservable window.
    A : :class:`float`
        Amplitude constant of the unobservable window.
    unobs_const : :class:`float`
        Baseline unobservable probability inside the tile annulus.
    r_ext_factor : :class:`float`
        Maximum radius in units of ``R1`` for the exterior window.
    n_jobs : :class:`int`
        Number of parallel jobs for the objective evaluation.

    Returns
    -------
    :class:`dict`
        Dictionary with keys:

        - ``pixid`` : pixel index
        - ``n_opt_tiles`` : number of tiles optimized
        - ``tiles_ra``, ``tiles_dec`` : initial tile centers
        - ``TILEID`` : tile IDs
        - ``new_tiles_ra``, ``new_tiles_dec`` : optimized centers
        - ``total_unobs_pdf_before``, ``total_unobs_pdf_after``
        - ``nit`` : optimizer iteration count
        - ``elapsed_s`` : wall-clock time in seconds
    """
    t0 = time.time()
    tile_idx = np.where(tiles["pixid"] == pixid)[0]
    tiles_ra = np.asarray(tiles["RA"][tile_idx], dtype=np.float64)
    tiles_dec = np.asarray(tiles["DEC"][tile_idx], dtype=np.float64)
    tiles_idx = np.asarray(tiles["TILEID"][tile_idx], dtype=np.int64)
    n_opt_tiles = len(tiles_ra)

    neigh_ipix = hp.get_all_neighbours(Nside, pixid, nest=False)
    neigh_ipix = neigh_ipix[neigh_ipix != -1]

    gal_idx = np.where(gal_cat["pixid"] == pixid)[0]
    gal_ra = np.asarray(gal_cat["RA"][gal_idx], dtype=np.float64)
    gal_dec = np.asarray(gal_cat["DEC"][gal_idx], dtype=np.float64)
    gal_unobs = np.ones(len(gal_ra), dtype=np.float64)

    gal_idx_ext = select_gal_idx_ext_near_pixel(
        gal_cat, neigh_ipix, pixid, Nside, nest=False
    )
    gal_ra_ext = np.asarray(gal_cat["RA"][gal_idx_ext], dtype=np.float64)
    gal_dec_ext = np.asarray(gal_cat["DEC"][gal_idx_ext], dtype=np.float64)
    gal_unobs_ext = np.ones(len(gal_ra_ext), dtype=np.float64) * unobs_3pass

    gal_ra_comb = np.concatenate([gal_ra, gal_ra_ext])
    gal_dec_comb = np.concatenate([gal_dec, gal_dec_ext])
    gal_unobs_comb = np.concatenate([gal_unobs, gal_unobs_ext])

    pix_ra_diff, pix_dec_diff = get_pixel_radec_diff_corners(Nside, pixid, nest=False)
    tile_ra_bound = pix_ra_diff / (n_opt_tiles / Npasses)
    tile_dec_bound = pix_dec_diff / (n_opt_tiles / Npasses)

    tiles_radec = np.column_stack([tiles_ra, tiles_dec])
    total_unobs_pdf = cal_total_unobs_pdf(
        tiles_radec.flatten(),
        gal_ra_comb,
        gal_dec_comb,
        gal_unobs_comb,
        R0,
        R1,
        alpha,
        A,
        unobs_const,
        r_ext_factor=r_ext_factor,
        n_jobs=n_jobs,
    )

    res = optimize_tile_pos_fun(
        tiles_ra,
        tiles_dec,
        tile_ra_bound,
        tile_dec_bound,
        gal_ra_comb,
        gal_dec_comb,
        gal_unobs_comb,
        R0,
        R1,
        alpha,
        A,
        unobs_const,
        r_ext_factor,
        n_jobs,
    )
    nit = res.nit   # number of iterations
    new_tiles_radec = res.x.reshape(-1, 2)
    new_tiles_ra, new_tiles_dec = new_tiles_radec[:, 0], new_tiles_radec[:, 1]

    return {
        "pixid": int(pixid),
        "n_opt_tiles": n_opt_tiles,
        "tiles_ra": tiles_ra,
        "tiles_dec": tiles_dec,
        "TILEID": tiles_idx,
        "new_tiles_ra": new_tiles_ra,
        "new_tiles_dec": new_tiles_dec,
        "total_unobs_pdf_before": float(total_unobs_pdf),
        "total_unobs_pdf_after": float(res.fun),
        "nit": nit,
        "elapsed_s": time.time() - t0
    }


def main():
    """Optimize tile positions over HEALPix pixels and write a FITS table.

    Loads the galaxy catalog and tile file, assigns HEALPix pixels,
    runs :func:`process_one_pixid` in parallel, and writes optimized
    tile RA/Dec plus unobservable PDF diagnostics to ``--odir``.
    """
    parser = ArgumentParser("Optimize tile positions for the given tile file")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs for parallel processing")
    parser.add_argument("--n_pix_jobs", type=int, default=1, help="Number of jobs for parallel processing of HEALPix pixels")
    parser.add_argument("--Npasses", type=int, default=3, help="Number of passes")
    parser.add_argument("--odir", type=str, default="./output/", help="Output directory")
    args = parser.parse_args()
    #n_jobs = 4
    n_jobs = args.n_jobs          # workers inside cal_total_unobs_pdf per pixel
    n_pix_jobs = args.n_pix_jobs      # parallel HEALPix pixels; keep n_pix_jobs * inner jobs <= CPU cores
    Npasses = args.Npasses
    odir = args.odir
    os.makedirs(odir, exist_ok=True)
    
    
    TILE_INNER_RADIUS_DEG = 0.1085       # Tile inner radius in degrees
    TILE_OUTER_RADIUS_DEG = 0.5968       # Tile outer radius in degrees


    R0 = TILE_INNER_RADIUS_DEG
    R1 = TILE_OUTER_RADIUS_DEG
    # set the parameters for the unobservable probability distribution function
    alpha = np.log(2.0)/np.log(2.5)  # relates to the setting that pdf=1 at r=2.5*R1
    print("alpha:", alpha)
    r_ext_factor = 2**(1.0/alpha)
    print("r_ext_factor:", r_ext_factor)

    unobs_const = 0.5  # based on previous test, the mean probability of targets (r<20.5) assigned by fibers is about 50% for one pass
    ##unobs_3pass = 0.5**Npasses
    unobs_3pass = unobs_const ** 4.0   # (~3./0.7, 70% sky coverage for the cluster masked sky)
    print("unobs_3pass:", unobs_3pass)

    A = (1.0-unobs_const)/(2.5**alpha-1.0)
    print(f"A={A}")

    rmagcut = 20.5
    ra_min = 0.0
    ra_max = 4.0
    dec_min = 0.0
    dec_max = 4.0

    ## load galaxy targets
    ifile = "./input/demo_4x4.npz"
    data_all = np.load(ifile)
    
    gal_par_cat = Table()
    gal_par_cat["RA"] = data_all["target_coord"][:, 0]
    gal_par_cat["DEC"] = data_all["target_coord"][:, 1]
    gal_par_cat["cluster_mask"] = data_all["cluster_mask"]
    ## select galaxies near the (projected) galaxy clusters' centers
    cluster_mask = (gal_par_cat["cluster_mask"]==1)
    gal_cat = gal_par_cat[cluster_mask]
    N_gal = len(gal_cat)
    print(f"Number of total galaxies after cluster center region mask: {N_gal}")

    # load tile infor
    tiles = Table()
    tiles["RA"] = data_all["tile_coord"][:, 0]
    tiles["DEC"] = data_all["tile_coord"][:, 1]
    tiles["PASS"] = data_all["tile_pass"]
    tiles["TILEID"] = np.arange(len(tiles))
    
    print("Input tiles:", tiles[0:5])
    
    mask = (tiles["PASS"] > 0)&(tiles["PASS"] < Npasses)
    tiles = tiles[mask]
    N_tiles = len(tiles)
    print(f"Number of total tiles with 0< PASS < {Npasses}: {N_tiles}")

    Nside = 16
    print(f"HEALPix Nside: {Nside}")
    # 1) Assign HEALPix pixel ID to every galaxy
    gal_cat["pixid"] = hp.ang2pix(
        Nside, gal_cat["RA"], gal_cat["DEC"], nest=False, lonlat=True
    ).astype(np.int64)
    # Assign HEALPix pixel ID to every tile
    tiles["pixid"] = hp.ang2pix(
        Nside, tiles["RA"], tiles["DEC"], nest=False, lonlat=True
    ).astype(np.int64)
    # 2) sort the pixel ID of galaxies in ascending order
    gal_cat = gal_cat[np.argsort(gal_cat["pixid"])]
    #gal_pixid_unique = np.unique(gal_cat["pixid"])

    # 3) sort the pixel ID of tiles in ascending order
    tiles = tiles[np.argsort(tiles["pixid"])]
    tile_pixid_unique = np.unique(tiles["pixid"])
    print("total unique pixels:", len(tile_pixid_unique))

    # run the optimization for each pixel
    inner_n_jobs = max(1, n_jobs // n_pix_jobs)

    pix_results = Parallel(n_jobs=n_pix_jobs, backend="loky")(
        delayed(process_one_pixid)(
            pixid,
            tiles,
            gal_cat,
            Nside,
            Npasses,
            unobs_3pass,
            R0,
            R1,
            alpha,
            A,
            unobs_const,
            r_ext_factor,
            inner_n_jobs,
        )
        for pixid in tile_pixid_unique
    )

    for out in pix_results:
        print(
            f"pixid={out['pixid']}: {out['n_opt_tiles']} tiles, "
            f"unobs {out['total_unobs_pdf_before']:.2f} -> {out['total_unobs_pdf_after']:.2f}, "
            f"nit={out['nit']}, "
            f"Optimization time={out['elapsed_s']:.1f}s"
        )
    # output the results: one FITS row per tile (not per pixel)
    pixids = []
    tileids = []
    ra_init = []
    dec_init = []
    ra_new = []
    dec_new = []
    unobs_before = []
    unobs_after = []
    opt_nit = []
    opt_elapsed = []

    for out in pix_results:
        n_tile = out["n_opt_tiles"]
        pixids.extend([out["pixid"]] * n_tile)
        tileids.extend(np.asarray(out["TILEID"], dtype=np.int64).tolist())
        ra_init.extend(np.asarray(out["tiles_ra"], dtype=np.float64).tolist())
        dec_init.extend(np.asarray(out["tiles_dec"], dtype=np.float64).tolist())
        ra_new.extend(np.asarray(out["new_tiles_ra"], dtype=np.float64).tolist())
        dec_new.extend(np.asarray(out["new_tiles_dec"], dtype=np.float64).tolist())
        unobs_before.extend([out["total_unobs_pdf_before"]] * n_tile)
        unobs_after.extend([out["total_unobs_pdf_after"]] * n_tile)
        opt_nit.extend([out["nit"]] * n_tile)
        opt_elapsed.extend([out["elapsed_s"]] * n_tile)

    tiles_results = Table(
        {
            "pixid": np.asarray(pixids, dtype=np.int64),
            "TILEID": np.asarray(tileids, dtype=np.int64),
            "RA": np.asarray(ra_init, dtype=np.float64),
            "DEC": np.asarray(dec_init, dtype=np.float64),
            "RA_NEW": np.asarray(ra_new, dtype=np.float64),
            "DEC_NEW": np.asarray(dec_new, dtype=np.float64),
            "UNOBS_PDF_BEFORE": np.asarray(unobs_before, dtype=np.float64),
            "UNOBS_PDF_AFTER": np.asarray(unobs_after, dtype=np.float64),
            "OPT_NIT": np.asarray(opt_nit, dtype=np.int64),
            "OPT_ELAPSED_S": np.asarray(opt_elapsed, dtype=np.float64),
        }
    )
    tiles_results.write(odir + f"/tiles_optimized_pass1_2.fits", overwrite=True)

    fig, ax = plt.subplots(dpi=120)
    
    ax.plot(gal_par_cat["RA"][cluster_mask], gal_par_cat["DEC"][cluster_mask], ",", color="gray", alpha=1.0)
    ax.plot(gal_par_cat["RA"][~cluster_mask], gal_par_cat["DEC"][~cluster_mask], ",", color="gray", alpha=0.5)
    ax.plot(tiles_results["RA"], tiles_results["DEC"], "o", label="default")
    ax.plot(tiles_results["RA_NEW"], tiles_results["DEC_NEW"], "x", label="optimized")
    ax.set_xlabel("RA [deg]", fontsize=14)
    ax.set_ylabel("DEC [deg]", fontsize=14)
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(odir + "/optimized_tiles.png")
    

    

if __name__ == "__main__":
    main()
