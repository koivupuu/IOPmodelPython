import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from iopmodel.misc import rnorm_bound, vec_to_mat
from iopmodel.global_variables import wavelen_IOP, ag_lib, coef_exp_ads, frac_phyto_lib, phyto_lib_aphs, phyto_lib_bphs, phyto_lib_cphs, phyto_lib_info

#### Detritus ####

def IOP_d_B22_C2(ISM,
                 Chl,
                 A=None,
                 G=None,
                 A_bd=None,
                 S_bd=None,
                 C_bd=None,
                 A_md=None,
                 S_md=None,
                 C_md=None,
                 qt_bd=0.5,
                 qt_md=0.5,
                 wavelen=None):
    """
    Calculate absorption and backscattering coefficients for IOP model B22_C2.

    This function computes the absorption (ad), backscattering (bd), and total attenuation (cd) coefficients
    for a given set of input parameters, using either provided coefficients or quantile-based sampling from
    empirical distributions. The function is designed for use in inherent optical property (IOP) modeling
    of aquatic systems.

    Parameters
    ----------
        ISM : float
            Inorganic suspended matter concentration.
        Chl : float
            Chlorophyll-a concentration.
        A : float, optional
            Power-law albedo parameter for backscattering (default: sampled from normal distribution).
        G : float, optional
            Power-law exponent for backscattering (default: sampled from truncated normal distribution).
        A_bd : float, optional
            Absorption coefficient parameter for background dissolved substances (optional).
        S_bd : float, optional
            Slope parameter for background dissolved substances (optional).
        C_bd : float, optional
            Constant parameter for background dissolved substances (optional).
        A_md : float, optional
            Absorption coefficient parameter for mineral dissolved substances (optional).
        S_md : float, optional
            Slope parameter for mineral dissolved substances (optional).
        C_md : float, optional
            Constant parameter for mineral dissolved substances (optional).
        qt_bd : float, optional
            Quantile for background dissolved substances coefficient sampling (default: 0.5).
        qt_md : float, optional
            Quantile for mineral dissolved substances coefficient sampling (default: 0.5).
        wavelen : array-like, optional
            Wavelengths (in nm) at which to compute the coefficients. If None, uses global `wavelen_IOP`.

    Returns
    -------
        dict
            Dictionary containing:
                - 'wavelen': array of wavelengths
                - 'ad': absorption coefficient array
                - 'bd': backscattering coefficient array
                - 'cd': total attenuation coefficient array
                - '_parm': dictionary of parameters used in the calculation
                - '_warn_neg_bd': bool, True if negative bd values were encountered and resampled

    Notes
    -----
        - If any of the coefficient parameters (A_bd, S_bd, C_bd, A_md, S_md, C_md) are not provided,
          the function samples them from empirical quantile distributions using `coef_exp_ads`.
        - If negative backscattering values are encountered, the function resamples coefficients
          and parameters until all values are non-negative.
        - Requires global variables: `coef_exp_ads`, `wavelen_IOP`, and function `rnorm_bound`.
    """

    if wavelen is None:
        wavelen = wavelen_IOP  # Assume global or passed separately

    use_qt = False

    if any(param is None for param in [A_bd, S_bd, C_bd, A_md, S_md, C_md]):
        use_qt = True
    else:
        coef_bd = {'A_bd': A_bd, 'S_bd': S_bd, 'C_bd': C_bd}
        coef_md = {'A_md': A_md, 'S_md': S_md, 'C_md': C_md}

    if use_qt:
        qt_bd = np.round(np.random.uniform(), 2) if qt_bd is None else qt_bd
        qt_md = np.round(np.random.uniform(), 2) if qt_md is None else qt_md

        def sample_coef(df, qt):
            """
            Interpolates parameter values ('A', 'S', 'C') at a given quantile using linear interpolation.

            Parameters:
                df (pandas.DataFrame): DataFrame containing columns 'quantile', 'A', 'S', and 'C'.
                qt (float): The quantile at which to interpolate the parameter values.

            Returns:
                dict: A dictionary with keys 'A', 'S', and 'C', where each value is the interpolated parameter at the specified quantile.
            """
            result = {}
            for param in ["A", "S", "C"]:
                x = df["quantile"]
                y = df[param]
                f = interp1d(x, y, fill_value="extrapolate")
                result[param] = float(f(qt))
            return result


        coef_bd_raw = sample_coef(coef_exp_ads[coef_exp_ads["comp"] == "abds"], qt_bd)
        coef_bd = {f"{k}_bd": v for k, v in coef_bd_raw.items()}

        coef_md_raw = sample_coef(coef_exp_ads[coef_exp_ads["comp"] == "amds"], qt_md)
        coef_md = {f"{k}_md": v for k, v in coef_md_raw.items()}

    # Absorption
    abds = coef_bd["A_bd"] * np.exp(-coef_bd["S_bd"] * (wavelen - 550)) + coef_bd["C_bd"]
    amds = coef_md["A_md"] * np.exp(-coef_md["S_md"] * (wavelen - 550)) + coef_md["C_md"]
    ad = Chl * abds + ISM * amds
    ad550 = float(interp1d(wavelen, ad)(550))

    # Coefficients for powerlaw
    A_mean, A_sd = -1.3390, 0.0618
    G_mean, G_sd = 0.3835, 0.1277

    if A is None or np.isnan(A):
        A = np.random.normal(A_mean, A_sd)
    if G is None or np.isnan(G):
        G = rnorm_bound(1, mean=G_mean, sd=G_sd, lo=0)

    Albedo_550 = A if A >= 0 else 1 - 10 ** A

    cd550 = ad550 / (1 - Albedo_550)
    cd = cd550 * (wavelen / 550) ** -G
    bd = cd - ad

    warn_neg_bd = False

    while np.any(bd < 0):
        if use_qt:
            print("Negative bd! Rational random values will be used!")
            warn_neg_bd = True

        qt_bd = np.round(np.random.uniform(), 2)
        qt_md = np.round(np.random.uniform(), 2)

        coef_bd_raw = sample_coef(coef_exp_ads[coef_exp_ads["comp"] == "abds"], qt_bd)
        coef_bd = {f"{k}_bd": v for k, v in coef_bd_raw.items()}

        coef_md_raw = sample_coef(coef_exp_ads[coef_exp_ads["comp"] == "amds"], qt_md)
        coef_md = {f"{k}_md": v for k, v in coef_md_raw.items()}

        abds = coef_bd["A_bd"] * np.exp(-coef_bd["S_bd"] * (wavelen - 550)) + coef_bd["C_bd"]
        amds = coef_md["A_md"] * np.exp(-coef_md["S_md"] * (wavelen - 550)) + coef_md["C_md"]
        ad = Chl * abds + ISM * amds
        ad550 = float(interp1d(wavelen, ad)(550))

        A = np.random.normal(A_mean, A_sd * 2)
        G = rnorm_bound(mean=G_mean, sd=G_sd * 2, lo=0)

        Albedo_550 = 1 - 10 ** A
        cd550 = ad550 / (1 - Albedo_550)
        cd = cd550 * (wavelen / 550) ** -G
        bd = cd - ad

    r = {
        "wavelen": wavelen,
        "ad": ad,
        "bd": bd,
        "cd": cd
    }

    parm = {
        "Chl": Chl,
        "ISM": ISM,
        "A_cd": A,
        "G_cd": G,
        "qt_bd": qt_bd,
        "qt_md": qt_md,
        "Albedo_550": Albedo_550,
        "A_bd": coef_bd["A_bd"],
        "S_bd": coef_bd["S_bd"],
        "C_bd": coef_bd["C_bd"],
        "A_md": coef_md["A_md"],
        "S_md": coef_md["S_md"],
        "C_md": coef_md["C_md"]
    }

    r["_parm"] = parm
    r["_warn_neg_bd"] = warn_neg_bd

    return r

#### Phytoplankton ####

def IOP_ph_B22_C2(Chl,
                  frac_phyto=None,
                  wavelen=None,
                  aphs=None,
                  bphs=None,
                  aphs_func=None,
                  **kwargs):
    """
    Calculate inherent optical properties (IOPs) for phytoplankton using the B22_C2 model.

    Parameters
    ----------
    Chl : float or array-like
        Chlorophyll-a concentration(s) [mg m^-3].
    frac_phyto : array-like, optional
        Fractional composition of phytoplankton groups. If None, random fractions are generated.
    wavelen : array-like, optional
        Wavelengths [nm] at which to compute IOPs. If None, uses default `wavelen_IOP`.
    aphs : array-like, optional
        Specific absorption coefficients for phytoplankton groups. If None, generated internally.
    bphs : array-like, optional
        Specific scattering coefficients for phytoplankton groups. If None, generated internally.
    aphs_func : callable, optional
        Function to generate or modify `aphs`. If None, uses `calc_aph676`.
    **kwargs
        Additional keyword arguments passed to `generate_aphs`.

    Returns
    -------
    dict
        Dictionary containing:
            - 'wavelen': array of wavelengths [nm]
            - 'aph': array of total phytoplankton absorption coefficients
            - 'bph': array of total phytoplankton scattering coefficients
            - 'cph': array of total phytoplankton attenuation coefficients
            - '_parm': dictionary of parameters used in the calculation
            - '_attr_aph676': optional attribute from `generate_aphs`
            - '_attr_ph_parm': optional attribute from `generate_aphs`

    Notes
    -----
    - The function interpolates specific IOPs to the desired wavelengths.
    - Phytoplankton group fractions are expanded to match the number of wavelengths.
    - Requires external functions: `generate_aphs`, `calc_aph676`, `rand_frac`, `vec_to_mat`, and `wavelen_IOP`.
    """

    if wavelen is None:
        wavelen = wavelen_IOP

    if frac_phyto is None:
        frac_phyto = np.array(list(rand_frac().values()))

    attr_aph676 = None
    attr_ph_parm = None

    if aphs is None or bphs is None:
        siop_ph = generate_aphs(Chl, aphs_fun=aphs_func or calc_aph676, **kwargs)
        aphs = siop_ph["aphs"]
        bphs = siop_ph["bphs"]
        attr_aph676 = siop_ph.get("_attr_aph676", None)
        attr_ph_parm = siop_ph.get("_parm", None)

    def interpolate_matrix(df, xout):
        df_long = df.melt(id_vars=[df.columns[0]], var_name="variable", value_name="value")
        x_col = df.columns[0]
        result = {
            var: interp1d(g[x_col].values, g["value"].values, fill_value="extrapolate")(xout)
            for var, g in df_long.groupby("variable")
        }
        interpolated = np.column_stack([result[var] for var in df.columns[1:]])
        return interpolated  # return as numpy matrix

    aphs_interp = interpolate_matrix(pd.DataFrame(aphs), wavelen)
    bphs_interp = interpolate_matrix(pd.DataFrame(bphs), wavelen)

    frac_mat = vec_to_mat(frac_phyto, n=aphs_interp.shape[0], axis=1)

    aphs_sum = np.sum(frac_mat * aphs_interp, axis=1)
    bphs_sum = np.sum(frac_mat * bphs_interp, axis=1)
    cphs_sum = aphs_sum + bphs_sum

    aph = Chl * aphs_sum
    bph = Chl * bphs_sum
    cph = Chl * cphs_sum

    r = {
        "wavelen": wavelen,
        "aph": aph,
        "bph": bph,
        "cph": cph
    }

    parm = {
        "Chl": Chl,
        "aphs": aphs_interp,
        "bphs": bphs_interp,
        "cphs": aphs_interp + bphs_interp,
        "frac_phyto": frac_phyto
    }

    r["_parm"] = parm
    r["_attr_aph676"] = attr_aph676
    r["_attr_ph_parm"] = attr_ph_parm

    return r


def calc_aph676(Chl, varyA=True):
    """
    Calculate the absorption coefficient of phytoplankton at 676 nm (aph676) as a function of chlorophyll concentration.

    Parameters
    ----------
    Chl : array-like or float
        Chlorophyll concentration(s) (mg/m^3).
    varyA : bool, optional (default=True)
        If True, randomly vary the coefficient 'A' within specified bounds using a log-normal distribution.

    Returns
    -------
    dict
        Dictionary containing:
            - "values": Computed aph676 values (same shape as `Chl`).
            - "A_aphs676": The coefficient 'A' used in the calculation.
            - "E_aphs676": The exponent 'E' used in the calculation.
            - "varyA": Boolean indicating if 'A' was varied.

    Notes
    -----
    If `varyA` is True, the coefficient 'A' is sampled from a log-normal distribution bounded by `A_lo` and `A_up`.
    For chlorophyll concentrations less than 1, a linear relationship is used; otherwise, a power-law relationship is applied.
    Requires `numpy` and a function `rnorm_bound` for bounded random sampling.
    """
    A = 0.0239
    A_sd = 0.1
    A_up = 0.0501
    A_lo = 0.0112
    E = 0.8938

    if varyA:
        logA = np.log10(A)
        logA_up = np.log10(A_up)
        logA_lo = np.log10(A_lo)
        A = 10 ** rnorm_bound(1, mean=logA, sd=A_sd, lo=logA_lo, up=logA_up)[0]

    Chl = np.asarray(Chl)
    r = np.where(Chl < 1, A * Chl ** 1, A * Chl ** E)

    result = {
        "values": r,
        "A_aphs676": A,
        "E_aphs676": E,
        "varyA": varyA
    }

    return result

def generate_aphs(
    Chl,
    albedo_ph676=[0.8952, 0.8952, 0.8800, 0.9000, 0.9000, 0.9562, 0.8485],
    gamma_ph676=[None, 0.00, 0.00, 0.00, 0.00, None, None],
    vary_cph=False,
    aphs_fun=calc_aph676,
    a_frac=[1, 1, 1, 1, 1, 1, 1],
    **kwargs
):
    """
    Generate absorption (aphs), backscattering (bphs), and attenuation (cphs) spectra for phytoplankton functional types
    based on chlorophyll concentration and optical properties.
    
    Parameters
    ----------
    Chl : float
        Chlorophyll-a concentration (mg/m^3).
    albedo_ph676 : list of float, optional
        Albedo values of phytoplankton at 676 nm for each functional type (default: [0.8952, ...]).
        Must be between 0 and 1.
    gamma_ph676 : list of float or None, optional
        Gamma values for phytoplankton attenuation normalized at 676 nm for each functional type.
        Values should be > -1. Use None for missing values.
    vary_cph : bool, optional
        If True, randomize albedo and gamma values within specified bounds (default: False).
    aphs_fun : callable, optional
        Function to compute aphs at 676 nm from Chl (default: calc_aph676).
    a_frac : list of float, optional
        Multiplicative factors for aphs for each functional type (default: [1, ...]).
        Additional keyword arguments passed to `aphs_fun`.
    
    Returns
    -------
    dict
        Dictionary containing:
            - 'aphs': DataFrame of absorption spectra (columns: functional types, rows: wavelengths)
            - 'bphs': DataFrame of backscattering spectra
            - 'cphs': DataFrame of attenuation spectra
            - 'parm': Dictionary of parameters used in the calculation
            - 'attr_aph676': Attributes from the aphs_fun result
            
    Raises
    ------
    ValueError
        If albedo_ph676 values are not between 0 and 1, or if gamma_ph676 values are less than -1.
        
    Notes
    -----
    - Uses reference data from Nardelli & Twardowski (2016) for albedo values.
    - Requires global variables: phyto_lib_info, phyto_lib_aphs, phyto_lib_cphs, phyto_lib_bphs, rnorm_bound, vec_to_mat, and numpy/pandas.
    """
    
    # Some albedo reference
    # Figure 8 (w532) of Nardelli, Schuyler C., and Michael S. Twardowski. “Assessing the
    # Link between Chlorophyll Concentration and Absorption Line Height at 676 Nm
    # over a Broad Range of Water Types.” Optics Express 24, no. 22 (October 31,
    # 2016): A1374. https://doi.org/10.1364/OE.24.0A1374.


    # 1. [Chl] -> aphs676
    # 2. [albedo] -> cphs676
    # 3. [Normalized aphs and cphs] -> bphs
    
    # Validate albedo_ph676 values
    if np.any(np.array(albedo_ph676) > 1) or np.any(np.array(albedo_ph676) < 0):
        raise ValueError("albedo of phytoplankton at 676 nm should be between 0 and 1")

    name_phyto = phyto_lib_info["ShortName"].tolist()

    # Map names to albedo_ph676 values
    albedo_ph676_dict = dict(zip(name_phyto, albedo_ph676))

    if vary_cph:
        albedo_ph676_new = {}
        for k, v in albedo_ph676_dict.items():
            albedo_ph676_new[k] = rnorm_bound(1, mean=v, sd=0.03, lo=None, up=0.99)[0]
        albedo_ph676_dict = albedo_ph676_new

    # Validate gamma_ph676 values (handle None/np.nan)
    gamma_ph676_array = np.array(gamma_ph676, dtype=np.float64)
    if np.any(gamma_ph676_array[~np.isnan(gamma_ph676_array)] < -1):
        raise ValueError("Gamma of phytoplankton attenuation normalized at 676 nm should be > -1")

    gamma_ph676_dict = dict(zip(name_phyto, gamma_ph676))

    if vary_cph:
        gamma_ph676_new = {}
        for k, v in gamma_ph676_dict.items():
            if v is None or np.isnan(v):
                gamma_ph676_new[k] = v  # skip NA values
            else:
                gamma_ph676_new[k] = rnorm_bound(1, mean=v, sd=0.01, lo=0.0, up=0.8)[0]
        gamma_ph676_dict = gamma_ph676_new

    # Get aph676 from function call
    aph676 = aphs_fun(Chl, **kwargs)

    # Extract attributes if applicable
    if hasattr(aph676, '__dict__'):
        attr_aph676 = aph676.__dict__
    else:
        attr_aph676 = {'aphs_fun': aphs_fun}

    # Extract numeric absorption at 676nm
    aphs_676_val = aph676['A_aphs676'] if isinstance(aph676, dict) else aph676

    aphs_676_new = aphs_676_val / Chl if Chl != 0 else 1

    aphs_df = phyto_lib_aphs
    aphs_mat = aphs_df.iloc[:, 1:].to_numpy()
    aphs_676 = aphs_df.loc[aphs_df['wv'] == 676].iloc[:, 1:].values.flatten()
    aphs_mat_norm = aphs_mat / vec_to_mat(aphs_676, aphs_mat.shape[0])

    aphs_mat_new = aphs_mat_norm * aphs_676_new

    cphs_df = phyto_lib_cphs
    cphs_mat = cphs_df.iloc[:, 1:].to_numpy()
    cphs_676 = cphs_df.loc[cphs_df['wv'] == 676].iloc[:, 1:].values.flatten()
    cphs_mat_norm = cphs_mat / vec_to_mat(cphs_676, cphs_mat.shape[0])

    wv = cphs_df['wv'].to_numpy()
    for i, key in enumerate(name_phyto):
        gamma = gamma_ph676_dict.get(key)
        if gamma is not None:
            cphs_mat_norm[:, i] = (676 / wv) ** gamma

    coef_lin = aphs_676_new / (1 - np.array(albedo_ph676))
    cphs_mat_new = cphs_mat_norm * vec_to_mat(coef_lin, cphs_mat.shape[0])
    bphs_mat_new = cphs_mat_new - aphs_mat_new

    aphs_mat_new *= vec_to_mat(a_frac, aphs_mat_new.shape[0])
    cphs_mat_new = aphs_mat_new + bphs_mat_new

    aphs = pd.DataFrame(aphs_mat_new, columns=name_phyto)
    aphs.insert(0, 'wv', phyto_lib_aphs['wv'])
    bphs = pd.DataFrame(bphs_mat_new, columns=name_phyto)
    bphs.insert(0, 'wv', phyto_lib_bphs['wv'])
    cphs = pd.DataFrame(cphs_mat_new, columns=name_phyto)
    cphs.insert(0, 'wv', phyto_lib_cphs['wv'])

    r = {
        'aphs': aphs,
        'bphs': bphs,
        'cphs': cphs
    }

    parm = {
        'aph676': aph676,
        'albedo_ph676': albedo_ph676,
        'gamma_ph676': gamma_ph676,
        'vary_cph': vary_cph,
        'a_frac': a_frac
    }

    r['parm'] = parm
    r['attr_aph676'] = attr_aph676

    return r

def rand_frac(case=2, Chl=0.5, r_up=None, r_lo=None):
    """
    Generate a random fraction distribution for phytoplankton types based on specified constraints.

    Parameters
    ----------
    case : int, optional
        Determines the default upper and lower bounds for the fractions if not provided.
        - case=1: Uses predefined bounds based on chlorophyll concentration (Chl).
        - case=2: Uses [0, 1] bounds for all types (default).
    Chl : float, optional
        Chlorophyll concentration (mg/m^3), used to adjust lower bounds in case 1. Default is 0.5.
    r_up : list of float, optional
        Upper bounds for each phytoplankton type. If None, bounds are set based on `case`.
    r_lo : list of float, optional
        Lower bounds for each phytoplankton type. If None, bounds are set based on `case`.

    Returns
    -------
    dict
        Dictionary mapping phytoplankton short names to their randomly generated fractions.

    Raises
    ------
    ValueError
        If the lengths of `r_up` or `r_lo` do not match the number of phytoplankton types,
        or if any lower bound is greater than its corresponding upper bound.

    Notes
    -----
    - The function samples from a phytoplankton fraction library (`frac_phyto_lib`) if possible,
      otherwise generates random fractions within the specified bounds.
    - The sum of the fractions is normalized to 1.
    - If the generated fractions do not satisfy the bounds after 50 iterations, the method switches
      from library sampling to uniform random generation.
    """

    nr, nc = frac_phyto_lib.shape
    phyto_names = list(phyto_lib_info['ShortName'])

    if r_up is not None and r_lo is not None:
        if len(r_up) != nc or len(r_lo) != nc:
            raise ValueError("Upper and lower bounds must match the number of phytoplankton types.")
        if any(lo > up for lo, up in zip(r_lo, r_up)):
            raise ValueError("Each lower bound must be <= corresponding upper bound.")

        prob_col = np.ones(nc)
        prob_row = np.ones(nr)

    else:
        if case == 1:
            r_up = [0.5, 0.5, 0.3, 0.5, 0.5, 0.1, 1.0]
            r_lo = [0.0] * nc

            if Chl < 0.05:
                r = [0] * (nc - 1) + [1]
                return dict(zip(phyto_names, r))

            elif 0.05 <= Chl < 0.2:
                r_lo = [0.0] * (nc - 1) + [0.5]

            elif 0.2 <= Chl < 1:
                r_lo = [0.0] * (nc - 1) + [0.2]

            elif Chl > 1:
                r_lo = [0.2] + [0.0] * (nc - 2) + [0.2]

            prob_col = np.ones(nc)
            prob_row = np.ones(nr)

        elif case == 2:
            r_up = [1.0] * nc
            r_lo = [0.0] * nc
            prob_col = np.ones(nc)
            prob_row = np.ones(nr)

    # Initial draw from library
    r_row = frac_phyto_lib.sample(1, weights=prob_row, axis=0).values[0]
    r = r_row[np.random.choice(nc, size=nc, replace=False, p=prob_col / np.sum(prob_col))]
    r = dict(zip(phyto_names, r))

    cond = True
    iter = 1
    from_lib = True

    while cond:
        if from_lib:
            r_row = frac_phyto_lib.sample(1, weights=prob_row, axis=0).values[0]
            col_indices = np.random.choice(nc, size=nc, replace=False, p=prob_col / np.sum(prob_col))
            r = r_row[col_indices]
        else:
            r = np.array([np.random.uniform(lo, up) for lo, up in zip(r_lo, r_up)])
            r = np.round(r / np.sum(r), 4)

        r_dict = dict(zip(phyto_names, r))
        iter += 1

        if all(r[i] <= r_up[i] for i in range(nc)) and all(r[i] >= r_lo[i] for i in range(nc)):
            cond = False

        if iter == 50 and not from_lib:
            cond = False

        if iter == 50:
            from_lib = False
            iter = 1

    return r_dict

#### CDOM ####

def IOP_cdom_B22_C2(ag440=None, S=None, wavelen=None):
    """
    Calculate the absorption coefficient of colored dissolved organic matter (CDOM) using the B22_C2 model.
    
    Parameters
    ----------
    ag440 : float, optional
        Absorption coefficient at 440 nm. If None or NaN, a value is sampled from a log-normal distribution.
    S : float, optional
        Slope parameter for the exponential decay. If None or NaN, a value is sampled from a normal distribution.
    wavelen : array-like
        Wavelength(s) at which to compute the absorption coefficient. Must be provided.
    Returns
    -------
    result : dict
        Dictionary containing:
            - "wavelen": The input wavelength(s).
            - "acdom": The computed CDOM absorption coefficient(s) at the given wavelength(s).
    params : dict
        Dictionary containing the parameters used:
            - "ag440": The absorption coefficient at 440 nm.
            - "S_cdom": The slope parameter used.
    Raises
    ------
    ValueError
        If `wavelen` is not provided.
    """
    
    if wavelen is None:
        raise ValueError("wavelen must be provided")

    # Sample ag440 if None or NaN
    if ag440 is None or (isinstance(ag440, float) and np.isnan(ag440)):
        ag440 = 10 ** np.random.normal(loc=-0.9425989, scale=0.4291600 / 2)

    # Sample S if None or NaN
    if S is None or (isinstance(S, float) and np.isnan(S)):
        S = np.round(np.random.normal(loc=0.017400895, scale=0.001395171 / 2), 4)

    acdom = ag440 * np.exp(-S * (np.array(wavelen) - 440))

    result = {
        "wavelen": wavelen,
        "acdom": acdom,
    }

    params = {"ag440": ag440, "S_cdom": S}

    return result, params


def IOP_cdom_B22_C2_lib(ag440, wavelen, ag_seed=1234):
    """
    Generates a CDOM absorption spectrum (acdom) at specified wavelengths using a randomly selected sample from a provided absorption library.

    Parameters:
        ag440 (float): The absorption coefficient at 440 nm.
        wavelen (array-like): Array of wavelengths at which to compute the absorption spectrum.
        ag_seed (int, optional): Random seed for reproducibility when selecting a sample from the library. Default is 1234.

    Returns:
        result (dict): Dictionary containing:
            - "wavelen": The input wavelengths (numpy array).
            - "acdom": The computed absorption spectrum at the specified wavelengths (numpy array).
            - "ag_seed": The random seed used for sample selection.
        params (dict): Dictionary containing the input parameter "ag440".

    Raises:
        ValueError: If the global variable 'ag_lib' (the absorption library DataFrame) is not provided.

    Notes:
        - The function expects a global variable 'ag_lib' to be defined, containing columns 'SampleID', 'wavelen', and 'value'.
        - If the minimum wavelength is less than 354 nm, the sample with SampleID "I080919" is excluded from selection.
        - For wavelengths >= 600 nm where interpolation is not possible, the absorption is set to 0.
    """

    if ag_lib is None:
        raise ValueError("ag_lib DataFrame must be provided")

    np.random.seed(ag_seed)

    wavelen = np.array(wavelen)

    # Select SampleID depending on min wavelength
    if wavelen.min() < 354:
        # Exclude SampleID "I080919"
        filtered_lib = ag_lib[ag_lib['SampleID'] != "I080919"]
        sample_id = np.random.choice(filtered_lib['SampleID'].unique())
        ag_norm_df = filtered_lib[filtered_lib['SampleID'] == sample_id][['wavelen', 'value']]
    else:
        sample_id = np.random.choice(ag_lib['SampleID'].unique())
        ag_norm_df = ag_lib[ag_lib['SampleID'] == sample_id][['wavelen', 'value']]

    # Interpolate ag_norm values at requested wavelengths
    ag_norm_interp = np.interp(wavelen, ag_norm_df['wavelen'], ag_norm_df['value'], left=np.nan, right=np.nan)

    # Replace NA values for wavelengths >= 600 with 0
    w_nan = np.where(np.isnan(ag_norm_interp) & (wavelen >= 600))[0]
    ag_norm_interp[w_nan] = 0

    acdom = ag440 * ag_norm_interp

    result = {
        "wavelen": wavelen,
        "acdom": acdom,
        "ag_seed": ag_seed
    }

    params = {"ag440": ag440}

    return result, params

#### Alternative bph functions ####

def specific_bph_GM83(Chl, wavelen, b0=0.3, n=0.62, m=1, wavelen0=550):
    """
    Calculate the specific backscattering coefficient for phytoplankton using the GM83 model.

    Parameters:
        Chl (float or array-like): Chlorophyll-a concentration (mg/m^3).
        wavelen (float or array-like): Wavelength (nm).
        b0 (float, optional): Reference backscattering coefficient at wavelen0 (default is 0.3).
        n (float, optional): Exponent for chlorophyll dependence (default is 0.62).
        m (float, optional): Exponent for wavelength dependence (default is 1).
        wavelen0 (float, optional): Reference wavelength (nm, default is 550).

    Returns:
        float or array-like: Specific backscattering coefficient for phytoplankton.
    """

    return b0 * Chl**n * (wavelen0 / wavelen)**m / Chl

def specific_bph_LM88(Chl, wavelen, b0=0.416, n=0.766, wavelen0=550):
    """
    Calculate the specific backscattering coefficient of phytoplankton using the Loisel & Morel (1988) model.

    Parameters
    ----------
    Chl : float or array-like
        Chlorophyll-a concentration (mg m^-3).
    wavelen : float or array-like
        Wavelength (nm).
    b0 : float, optional
        Empirical coefficient (default is 0.416).
    n : float, optional
        Exponent for chlorophyll dependence (default is 0.766).
    wavelen0 : float, optional
        Reference wavelength (nm, default is 550).

    Returns
    -------
    float or array-like
        Specific backscattering coefficient of phytoplankton (m^2 mg^-1).

    Notes
    -----
    The exponent 'm' depends on the chlorophyll concentration:
        - If Chl >= 2, m = 0
        - If Chl < 2, m = 0.5 * (log10(Chl) - 0.3)
    """

    # m depends on Chl value:
    if Chl >= 2:
        m = 0
    else:
        m = 0.5 * (np.log10(Chl) - 0.3)
    return b0 * Chl**n * (wavelen / wavelen0)**m / Chl

def specific_bph_G99(Chl, wavelen, b0=0.5, n=0.62, m=-0.00113, i=1.62517, wavelen0=550):
    """
    Calculate the specific backscattering coefficient for phytoplankton using the G99 model.

    Parameters:
        Chl (float or array-like): Chlorophyll-a concentration.
        wavelen (float or array-like): Wavelength (in nm).
        b0 (float, optional): Reference backscattering coefficient at wavelen0. Default is 0.5.
        n (float, optional): Exponent for chlorophyll dependence. Default is 0.62.
        m (float, optional): Slope parameter for wavelength dependence. Default is -0.00113.
        i (float, optional): Intercept parameter for wavelength dependence. Default is 1.62517.
        wavelen0 (float, optional): Reference wavelength (in nm). Default is 550.

    Returns:
        float or array-like: Specific backscattering coefficient for phytoplankton at the given wavelength and chlorophyll concentration.

    References:
        - Loisel, H., & Morel, A. (1998). Light scattering and chlorophyll concentration in case 1 waters: A reexamination. Limnology and Oceanography, 43(5), 847-858.
        - Gordon, H. R., et al. (1999). A semianalytic radiance model of ocean color. Journal of Geophysical Research: Oceans, 104(C3), 5011-5028.
    """

    numerator = m * wavelen + i
    denominator = m * wavelen0 + i
    return b0 * (numerator / denominator) * Chl**n / Chl

#### Alternative aph functions ####

def specific_aph_B04(Chl=1, a=0.0654, b=0.7280, frac_phyto=None, wavelen=None, phytodive_iop=None):
    """
    Calculate the specific absorption coefficient of phytoplankton at given wavelengths using the B04 model.
    
    Parameters
    ----------
    Chl : float or array-like, optional
        Chlorophyll-a concentration (default is 1).
    a : float, optional
        Empirical coefficient for the B04 model (default is 0.0654).
    b : float, optional
        Empirical exponent for the B04 model (default is 0.7280).
    frac_phyto : array-like, optional
        Fractional contribution of each phytoplankton group (default is equal fractions).
    wavelen : array-like
        List or array of wavelengths (in nm) for which to calculate absorption.
    phytodive_iop : dict
        Dictionary containing inherent optical properties (IOPs) from PhytoDOIve, 
        specifically must contain 'phs_n440' with an 'aphs' DataFrame including 'wv' and absorption columns.
        
    Returns
    ----------
    aphs_sum : numpy.ndarray
        Array of specific absorption coefficients at the specified wavelengths.
    """

    if frac_phyto is None:
        frac_phyto = np.ones(7) / 7  # equal fractions
    
    # Extract absorption data for wavelengths of interest
    aphs_df = phytodive_iop['phs_n440']['aphs']
    aphs_df = aphs_df[aphs_df['wv'].isin(wavelen)]
    aphs_mat = aphs_df.drop(columns=['wv']).to_numpy()
    
    frac_mat = np.tile(frac_phyto, (aphs_mat.shape[0], 1))
    
    aphs_440 = a * Chl**b / Chl  # scalar or array
    
    aphs_sum = np.sum(frac_mat * aphs_mat, axis=1) * aphs_440
    
    return aphs_sum


def specific_aph_B98(Chl=1, wavelen=None, B98_AE_midUVabs=None):
    """
    Calculates the specific absorption coefficient of phytoplankton (aph*) using the Bricaud et al. (1998) model.
    Parameters:
        Chl (float, optional): Chlorophyll-a concentration (mg m^-3). Default is 1.
        wavelen (array-like): Wavelength(s) (nm) at which to compute aph*. Should be within or near the range of AE['wavelen'].
        B98_AE_midUVabs (dict): Dictionary containing Bricaud et al. (1998) model parameters with keys:
            - 'wavelen': array-like, wavelengths (nm) of the model parameters.
            - 'A': array-like, model parameter A at each wavelength.
            - 'E': array-like, model parameter E at each wavelength.
    Returns:
        numpy.ndarray: Specific absorption coefficient of phytoplankton (aph*) at the specified wavelengths (m^2 mg^-1).
    """

    AE = B98_AE_midUVabs
    
    # Interpolate A and E to desired wavelengths, using fill_value='extrapolate'
    interp_A = interp1d(AE['wavelen'], AE['A'], bounds_error=False, fill_value="extrapolate")
    interp_E = interp1d(AE['wavelen'], AE['E'], bounds_error=False, fill_value="extrapolate")
    
    A_vals = interp_A(wavelen)
    E_vals = interp_E(wavelen)
    
    aphs = A_vals * Chl**E_vals / Chl
    
    return aphs