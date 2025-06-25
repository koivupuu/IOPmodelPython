import numpy as np
import pandas as pd
import importlib.resources as pkg_resources

def load_csv(filename):
    data_path = pkg_resources.files('iopmodel.data') / filename
    with data_path.open('r') as f:
        return pd.read_csv(f)

wavelen_IOP = np.arange(400, 801, 1)

coef_exp_ads = load_csv("coef_exp_ads.csv")
phyto_lib_info = load_csv("Phyto_lib_info.csv")
frac_phyto_lib = load_csv("frac_phyto_lib.csv")
phyto_lib_aphs = load_csv("Phyto_lib_aphs.csv")
phyto_lib_bphs = load_csv("Phyto_lib_bphs.csv")
phyto_lib_cphs = load_csv("Phyto_lib_cphs.csv")
ag_lib = load_csv("ag_hat_lib.csv")
WOPP_computed_refri_T27_S0_180_4000nm = load_csv("WOPP_computed_refri_T27_S0_180_4000nm.csv")
WOPP_purewater_abs_coefficients = load_csv("WOPP_purewater_abs_coefficients.csv")