# Combine information from multiple .csv files.
# Intended to be run on the cluster.

import numpy as np
import pandas as pd
import glob


files = glob.glob("velocities/*csv")
df = pd.read_csv(files[0])

for i in range(1, len(files)):
    df = pd.concat([df, pd.read_csv(files[i])])

df.to_csv("all_stars.csv")
