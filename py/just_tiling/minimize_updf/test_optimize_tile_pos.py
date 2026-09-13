import os
import time
import numpy as np
import healpy as hp
from astropy.table import Table
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from optimize_tile_pos import process_one_pixid


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
