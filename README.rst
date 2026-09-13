============
just_tiling
============

Introduction
============

This repository is intended for the scheduling of tiles for JUST galaxy/galaxy cluster survey.

.. _JUST: https://just.sjtu.edu.cn/EN/about.php



Installation
============

The package requires Python 3.8 or later. The easiest way to install it is
directly from GitHub using ``pip``:

.. code-block:: bash

    pip install git+https://github.com/zjdingastro/just_tiling.git

Install from source
-------------------

To install a local editable copy for development, clone the repository and
install it with ``pip``:

.. code-block:: bash

    git clone https://github.com/zjdingastro/just_tiling.git
    cd just_tiling
    pip install -e .

Install dependencies
--------------------

If you only want to install the runtime dependencies (for example, before
running scripts directly from the repository), use the provided
``requirements.txt``:

.. code-block:: bash

    pip install -r requirements.txt

The dependencies include ``numpy``, ``matplotlib``, ``healpy``, ``astropy``,
``scipy``, and ``joblib``.


Running the hyperuniform tiling example
=======================================

The hyperuniform tiling example is located at
``py/just_tiling/hyperuniform/hyperuniform_tiling.py``. It uses hardcoded
paths to read from and write to ``./testdata/`` in the same directory, so run
it from ``py/just_tiling/hyperuniform/``:

.. code-block:: bash

    cd py/just_tiling/hyperuniform
    python hyperuniform_tiling.py

The script reads the galaxy catalog
``testdata/lightcone_ra_0_90_dec_0_90_rmagcut20.5_cluster_mask.fits`` and
writes the initial, intermediate, and final tile distributions as ``.npy``
files in ``testdata/``. Edit the ``catalogfile`` and ``outputdir`` variables
at the top of the script to use different input or output paths.


Running the minimize updf example
=================================

The tile-position optimization example is located at
``py/just_tiling/minimize_updf/test_optimize_tile_pos.py``. It expects the
demo input file ``input/demo_4x4.npz`` in the same directory, so run it from
``py/just_tiling/minimize_updf/``:

.. code-block:: bash

    cd py/just_tiling/minimize_updf
    python test_optimize_tile_pos.py

By default the script writes its outputs to ``./output/``. You can override
this and other options from the command line:

.. code-block:: bash

    python test_optimize_tile_pos.py --odir ./my_output --n_jobs 4 --n_pix_jobs 2 --Npasses 3

Available options:

- ``--n_jobs``: total number of parallel workers available to the optimizer.
- ``--n_pix_jobs``: number of HEALPix pixels processed in parallel
  (inner workers per pixel will be ``n_jobs // n_pix_jobs``).
- ``--Npasses``: number of survey passes used to set tile move bounds.
- ``--odir``: output directory for the FITS table and diagnostic plot.
