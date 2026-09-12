import numpy as np
from astropy.io import fits
import healpy as hp


tile_outer = 0.5968   # deg
tile_inner = 0.1085   # deg
 

### read galaxy catalogue

hdul = fits.open('lightcone_ra_0_90_dec_0_90_rmagcut20.5_cluster_mask.fits')
header=hdul[1].header
data=hdul[1].data

ra=data['ra']
#print(ra)
dec=data['dec']
cluster=data['cluster_mask']

ngaltot = len(ra)

print('number of galaxies: %i'%ngaltot)
print('number of cluster nearby galaxies: %i'%sum(cluster))



### set mask

nside = 256
npix = hp.nside2npix(nside)
print(f"总像素数: {npix}")

resolution=hp.pixelfunc.nside2resol(nside, arcmin=True)
print(f"分辨率: {resolution:8.2f} arcmin")

theta=np.deg2rad(90-dec)
phi=np.deg2rad(ra)

pix_indices = hp.ang2pix(nside, theta, phi, nest=False)
counts_map = np.bincount(pix_indices, minlength=npix)
mask = counts_map == 0
visible = ~mask

numbermask = mask.sum()
numbervisible = visible.sum()
print(f"number of mask pixel: {numbermask}")
print(f"number of visible pixel: {numbervisible}")


### galaxy number density map

galdensity = ngaltot/numbervisible
print(f"galaxy number density per pixel: {galdensity}")

massmap = np.zeros(npix)
massmap[visible] = galdensity/counts_map[visible]
massmap[mask] = hp.pixelfunc.UNSEEN


### downsample the galaxy catalog to make initial tile distribution

numtile=12000

dec0 = 0
dec1 = 90
ra0 = 0
ra1 = 90

trim_mask = (
      ( ra > ra0 )
    & ( ra < ra1 )        
    & ( dec > dec0 )
    & ( dec < dec1 )
    )

ngal_for_tile = np.sum(trim_mask)
print(f"galaxy catalog is trimmed from {ngaltot} to {ngal_for_tile} to avoid boundary")

dec_mask = dec[trim_mask]
ra_mask = ra[trim_mask]

fraction=numtile/ngal_for_tile
print(f"downsample fraction = {fraction:14.10f}")

tmpindex=np.linspace(0,int(ngal_for_tile-1),int(ngal_for_tile),dtype=int)

rng=np.random.default_rng(seed=42)
lucky=rng.choice(tmpindex,size=numtile,replace=False)

theta2=np.deg2rad(90-dec_mask[lucky])
phi2=np.deg2rad(ra_mask[lucky])

pix_indices2 = hp.ang2pix(nside, theta2, phi2, nest=False)

mass2 = massmap[pix_indices2]

massmin=mass2.min()
massmax=mass2.max()
print(massmin,massmax)
print(f"average tile mass: {np.sum(mass2)/len(mass2)}")




