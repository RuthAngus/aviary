# aviary
Inferring ages from velocity dispersions


AVIARY
======

avr.py
------
Code for applying the Yu and Liu AVR.

dispersion.py
-------------
Various tools for calculating the velocity dispersions of stars.

velocities.py
-------------
Some basic functions for calculating velocities from gaia data
using astropy. Includes definitions of Solar velocity constants.

velocity_pm_conversion.py
-------------------------
Some unit functions for coordinate conversions

pymc3_functions_one_star.py
---------------------------
Functions needed to do coordinate transformations
in a pymc3-friendly framework.
Calls functions in velocities.

inference.py
------------
Contains the prior and likelihood function for inferring
velocities.
Calls functions in pymc3_functions_one_star, velocities and
velocity_pm_conversion.
