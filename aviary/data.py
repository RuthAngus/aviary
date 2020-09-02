import pandas as pd
import pkg_resources

def kepler_data():
    f = pkg_resources.resource_filename(__name__, "mc_san_gaia_lam.csv")
    df = pd.read_csv(f)
    return df
