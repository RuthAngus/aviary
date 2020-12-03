# aviary
Inferring ages from velocity dispersions


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


CODE
====

*data.py*: Code for assembling data catalogs, used to calculate Kepler
velocities and to construct the prior.
This catalog is called by aviary/pymc3_functions_one_star:
mc_san_gaia_lam.csv.

DATA
====

Files used in data.py:

*kepler_dr2_1arcsec.fits*: Megan's crossmatched catalog

*santos.csv*: Rotation periods from Santos et al.

*KeplerRot-LAMOST.csv*: Jason Curtis' LAMOST file.

*Ruth_McQuillan_Masses_Out.csv*: Masses from Travis Berger.
