# aviary
Inferring ages from velocity dispersions and fitting a GP gyrochronology model.

To calculate velocities:
----------------------
*ssh into cluster and navigate to aviary_tests/aviary.
*Create a new run_.py file and a new pymc3_multi_star.py or edit the get_velocities_general.py.
*Change the upper and lower indices in the run file, depending on how many stars you want to run on.
*Change the directory and data file in code/get_velocities_general.py
*The data file must contain ra, ra_error, dec, dec_error, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error.
*Create a new .sh file and do module load slurm, sbatch <.sh file>. 
*To watch progress: tail -f slurm-.out

To calculate ages:
-----------------
Training the model:
run gp_fit.py

Inferring an age: 
Currently run Age_posterior in cv_routines.py

AVIARY
======

*gp_fit.py*: Fitting Gaussian process model to cluster and kinematic
data.

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

*notebooks/cv_routines.py*: code needed for performing cross validation.
Contains code for age posteriors and making the main plot, doesn't have
code for fitting GP model, that's in aviary/gp_fit.py.


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

Calibration data:

*dbf12.csv*: Jason's file of cluster data (modified from original .txt
version).

*Gyro_Benchmark-Field_Stars.csv*: Jason's file of benchmark stars.

*mcquillan_kinematic_data.csv*: My kinematic ages for Kepler stars.

*Gyrokinage2020_Prot.csv*: Lucy's kinematic ages.


NOTEBOOKS
=========

*Add_low_SN*: Experimenting with adding in the low S/N rotators.

*Cross_Validation*: using cross validation to test the quality of the fit.

*Get_cluster_scatter*: Calculate the scatter in the clusters to estimate errorbars.

*GP_fit*: a from-scratch version of the GP fit. Currently looks pretty good.

*Nice_plots*: Making a plot of kinematic data with cluster data for talks.

*Comparison_with_Spada*: Comparing model with the model of Spada & Lanzafame (2019).
Didn't get very far, but has code to load pickled results.

*GP_demo_hackszors*: mucking around with the GP fit.

*GP_demo*: Fitting the GP gyro relation.

*GP_fit_MEarth*: Fitting the GP gyro relation with MEarth data.

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