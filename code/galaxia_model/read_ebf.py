import ebf
import pandas as pd
from astropy.table import Table

fname = "kepler.ebf"

kepler = ebf.read(fname)
print(kepler.keys())

vx = ebf.read(fname, '/vx')
vy = ebf.read(fname, '/vy')
vz = ebf.read(fname, '/vz')
px = ebf.read(fname, '/px')
py = ebf.read(fname, '/py')
pz = ebf.read(fname, '/pz')
glon = ebf.read(fname, '/glon')
glat = ebf.read(fname, '/glat')
age = ebf.read(fname, '/age')
feh = ebf.read(fname, '/feh')
alpha = ebf.read(fname, '/alpha')
smass = ebf.read(fname, '/smass')
mact = ebf.read(fname, '/mact')
teff = ebf.read(fname, '/teff')
lum = ebf.read(fname, '/lum')
grav = ebf.read(fname, '/grav')
ubv_v = ebf.read(fname, '/ubv_v')
center = ebf.read(fname, '/center')
print(center)
# mbol = ebf.read('kepler.ebf', '/mbol')

k = pd.DataFrame(dict({
    "vx": vx,
    "vy": vy,
    "vz": vz,
    "px": px,
    "py": py,
    "pz": pz,
    "glon": glon,
    "glat": glat,
    "age": age,
    "feh": feh,
    "alpha": alpha,
    "smass": smass,
    "mact": mact,
    "teff": teff,
    "lum": lum,
    "grav": grav,
    "ubv_v": ubv_v,
    # "center": center,
    # "mbol": mbol,
}))

k.to_csv("galaxia1.csv")


# mag0 = ebf.read('kepler.ebf', '/')
# mag1 = ebf.read('kepler.ebf', '/')
# mag2 = ebf.read('kepler.ebf', '/')
# rad = ebf.read('kepler.ebf', '/')
# popid = ebf.read('kepler.ebf', '/')
# satid = ebf.read('kepler.ebf', '/')
# fieldid = ebf.read('kepler.ebf', '/')
# partid = ebf.read('kepler.ebf', '/')
# log = ebf.read('kepler.ebf', '/')
# photosys_band = ebf.read('kepler.ebf', '/')
# exbv_schlegel = ebf.read('kepler.ebf', '/')
# exbv_schlegel_inf = ebf.read('kepler.ebf', '/')
# exbv_solar = ebf.read('kepler.ebf', '/')
# mtip = ebf.read('kepler.ebf', '/')
