# aviary
Inferring ages from velocity dispersions

NOTEBOOKS
=========

*Aperiodoc*: exploring aperiodic stars.

*Are_Asteroseismic_Stars_Aperiodic*: More exploration of aperiodic stars.

*Better_Prior*: Testing the prior.

*Bokeh*: making main Bokeh plots.

*Existing_RVs*: Exploring stats of existing RVs.

*Full_age_comparison*: comparing kinematic ages with benchmark ages.

*Multi-dimensional*: More extensive work on comparisons.

*v_comparison*: Generating the velocity comparison figure for the paper.

*XD*: Applying XD to isochrone data.

*Yu_Liu_AVR*: Coding up Yu and Liu AVR.

AVIARY
======

*avr.py*:
Code for applying the Yu and Liu AVR.

*dispersion.py*:
Various tools for calculating the velocity dispersions of stars.

*velocities.py*:
Some basic functions for calculating velocities from gaia data
using astropy. Includes definitions of Solar velocity constants.

*velocity_pm_conversion.py*:
Some unit functions for coordinate conversions

*pymc3_functions_one_star.py*:
Functions needed to do coordinate transformations
in a pymc3-friendly framework.
Calls functions in velocities.

*inference.py*: Contains the prior and likelihood function for inferring
velocities.
Calls functions in pymc3_functions_one_star, velocities and
velocity_pm_conversion.

OTHER CODE
====

*code/data.py*: Code for assembling data catalogs, used to calculate Kepler
velocities and to construct the prior.
This catalog is called by aviary/pymc3_functions_one_star:
mc_san_gaia_lam.csv.

DATA
====

Files used in inference.py to construct a prior:
gaia_mc5_velocities.csv

This file is created in ~/projects/old_stuff/aviary///code/calculate_vxyz.py
from gaia_mc5.csv which is created in...

Files used in data.py:

*kepler_dr2_1arcsec.fits*: Megan's crossmatched catalog

*santos.csv*: Rotation periods from Santos et al.

*KeplerRot-LAMOST.csv*: Jason Curtis' LAMOST file.

*Ruth_McQuillan_Masses_Out.csv*: Masses from Travis Berger.

*Table_1_Periodic.txt*: Rotation periods from McQuillan 2014.
