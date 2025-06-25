import numpy as np
from scipy.interpolate import interp1d
from iopmodel.global_variables import WOPP_computed_refri_T27_S0_180_4000nm, WOPP_purewater_abs_coefficients


def WOPP(Temp=25, Sal=10, wavelen=None, aw_version=3):
    """
    Calculate the optical properties of pure water or seawater as a function of temperature, salinity, and wavelength.
    
    Parameters
    ----------
    Temp : float, optional
        Temperature in degrees Celsius. Default is 25.
    Sal : float, optional
        Salinity in PSU (Practical Salinity Units). Default is 10.
    wavelen : array-like
        Wavelengths (in nm) at which to compute the optical properties. Must be provided.
    aw_version : int, optional
        Version of the pure water absorption coefficients to use. Must be 1, 2, or 3. Default is 3.
    
    Returns
    -------
    dict
        Dictionary containing:
            'wavelen' : array-like
                Wavelengths (nm) at which properties are computed (matches input).
            'a' : array-like
                Absorption coefficients (1/m) at the specified wavelengths.
            'b' : array-like
                Scattering coefficients (1/m) at the specified wavelengths.
            'Temp' : float
                Temperature used in the calculation (deg C).
            'Sal' : float
                Salinity used in the calculation (PSU).
            'nw' : array-like
                Refractive index of water at the specified wavelengths.
            'aw_version' : int
                Version of the absorption coefficients used.
                
    Raises
    ------
    ValueError
        If `wavelen` is not provided or if `aw_version` is not 1, 2, or 3.
        
    Notes
    -----
    This function relies on external data tables for refractive index and absorption coefficients.
    The output is interpolated to match the user-supplied wavelengths.
    """
    if wavelen is None:
        raise ValueError("wavelen must be provided")

    aw = min(wavelen)
    ew = max(wavelen)
    S = Sal
    Tc = Temp

    def read_refri_std():
        """
        Reads and returns a subset of the refractive index standard data within a specified wavelength range.
        The function selects data from the global variable `WOPP_computed_refri_T27_S0_180_4000nm` where the wavelength (`wl`)
        is between `(aw - 1)` and `(ew + 1)`. It extracts the corresponding wavelength (`wl`), refractive index (`refri`),
        and computes the refractive index error (`refri_err`) based on the wavelength:
            - For wavelengths < 700 nm, error is set to 0.00003.
            - For wavelengths >= 700 nm, error is set to 0.0005.
        Returns:
            dict: A dictionary containing:
                - 'wl' (np.ndarray): Array of selected wavelengths.
                - 'refri' (np.ndarray): Array of refractive index values for the selected wavelengths.
                - 'refri_err' (np.ndarray): Array of refractive index errors for the selected wavelengths.
        Note:
            The variables `aw` and `ew` must be defined in the scope where this function is called.
            The global variable `WOPP_computed_refri_T27_S0_180_4000nm` must be available and structured as a DataFrame
            with columns 'wl' and 'refri'.
        """
        d = WOPP_computed_refri_T27_S0_180_4000nm
        
        idx = (d['wl'] >= (aw - 1)) & (d['wl'] <= (ew + 1))
        wl = d.loc[idx, 'wl'].to_numpy()
        refri = d.loc[idx, 'refri'].to_numpy()
        
        # Initialize refri_err array
        refri_err = np.zeros(len(d))
        refri_err[d['wl'] < 700] = 0.00003
        refri_err[d['wl'] >= 700] = 0.0005
        
        refri_err = refri_err[idx]
        
        return {'wl': wl, 'refri': refri, 'refri_err': refri_err}

    def absorption(fn=None, version=aw_version):
        """
        Calculates the absorption coefficients and associated errors for pure water based on wavelength, temperature, and salinity.

        Parameters:
            fn (str, optional): Filename or identifier for data source (currently unused).
            version (int): Version of the absorption coefficients to use. Must be 1, 2, or 3.

        Returns:
            dict: A dictionary containing:
                - 'wl' (np.ndarray): Array of wavelengths used in the calculation.
                - 'abso' (np.ndarray): Calculated absorption coefficients for each wavelength.
                - 'a_err' (np.ndarray): Estimated errors for the absorption coefficients.

        Raises:
            ValueError: If the provided version is not 1, 2, or 3.

        Notes:
            - Uses global variables: WOPP_purewater_abs_coefficients, aw, ew, Tc, S, aw_version.
            - The calculation is performed for wavelengths in the range (aw-1) to (ew+1).
            - Absorption is adjusted for temperature (Tc) and salinity (S) relative to reference values (T_ref=20.0°C, S_ref=0.0 PSU).
        """
        # Validate version
        if version not in [1, 2, 3]:
            raise ValueError("Version should be 1, 2, or 3")
        version_sel = version

        # Filter rows where version == version_sel
        d = WOPP_purewater_abs_coefficients[WOPP_purewater_abs_coefficients['version'] == version_sel]

        # Select indices where wl is in the range (aw-1) to (ew+1)
        idx = d.index[(d['wl'] >= (aw - 1)) & (d['wl'] <= (ew + 1))]

        d = d.loc[idx]

        wl = d['wl'].values
        abso_ref = d['a0'].values
        S_coeff = d['S_coeff'].values
        T_coeff = d['T_coeff'].values
        a0_err = d['a0_err'].values
        S_err = d['S_err'].values
        T_err = d['T_err'].values

        T_ref = 20.0
        S_ref = 0.0

        # Calculate absorption and error arrays
        abso = abso_ref + (Tc - T_ref) * T_coeff + (S - S_ref) * S_coeff
        a_err = a0_err + (Tc - T_ref) * T_err + (S - S_ref) * S_err

        return {'wl': wl, 'abso': abso, 'a_err': a_err}


    def scattering(wl, std_RI, theta=90):
        """
        Calculate the volume scattering properties of seawater due to density and concentration fluctuations.

        Parameters
        ----------
        wl : array-like or float
            Wavelength(s) in nanometers at which to compute scattering.
        std_RI : float or array-like
            Standard refractive index input for seawater.
        theta : float, optional
            Scattering angle in degrees (default is 90).

        Returns
        -------
        dict
            Dictionary containing:
                'betasw' : ndarray
                    Volume scattering function at specified angle(s).
                'beta90sw' : ndarray
                    Volume scattering at 90 degrees.
                'bsw' : ndarray
                    Total scattering coefficient.
                'err_betasw' : ndarray
                    Estimated error (2%) for 'betasw'.
                'err_beta90sw' : ndarray
                    Estimated error (2%) for 'beta90sw'.
                'err_bsw' : ndarray
                    Estimated error (2%) for 'bsw'.
                'nsw' : ndarray or float
                    Refractive index of seawater at given wavelength(s).

        Notes
        -----
        This function requires several external functions and variables to be defined in the scope:
        - refractive_index(std_RI)
        - BetaT()
        - rhou_sw()
        - dlnasw_ds()
        - PMH(nsw)
        - S (salinity)
        - Tc (temperature in Celsius)
        """
        delta = 0.039  # depolarization ratio

        Na = 6.0221417930e23   # Avogadro's number
        Kbz = 1.3806503e-23    # Boltzmann constant
        Tk = Tc + 273.15       # absolute temperature in K
        M0 = 18e-3             # molecular weight of water (kg/mol)

        rad = np.deg2rad(theta)  # convert angle to radians

        res_refri = refractive_index(std_RI)
        nsw = res_refri['nsw']          # seawater refractive index at wl
        dnds = res_refri['dnswds']      # partial derivative w.r.t salinity

        # Isothermal compressibility - function returns scalar or array matching wl
        IsoComp = BetaT()

        # density of seawater in kg/m3
        density_sw = rhou_sw()

        # partial derivative of ln(water activity) wrt salinity
        dlnawds = dlnasw_ds()

        # density derivative of refractive index from PMH model (array for wl)
        DFRI = PMH(nsw)

        # Convert wavelengths from nm to meters for scattering calculations
        wl_m = wl * 1e-9

        # Volume scattering at 90 degrees due to density fluctuations
        beta_df = (np.pi**2 / 2) * (wl_m ** -4) * Kbz * Tk * IsoComp * (DFRI ** 2) * (6 + 6 * delta) / (6 - 7 * delta)

        # Volume scattering at 90 degrees due to concentration fluctuations
        flu_con = S * M0 * (dnds ** 2) / density_sw / (-dlnawds) / Na
        beta_cf = 2 * (np.pi ** 2) * (wl_m ** -4) * (nsw ** 2) * flu_con * (6 + 6 * delta) / (6 - 7 * delta)

        beta90sw = beta_df + beta_cf

        bsw = (8 * np.pi / 3) * beta90sw * (2 + delta) / (1 + delta)

        # For angles, create a matrix of volume scattering
        betasw = np.outer(beta90sw, 1 + (np.cos(rad) ** 2) * (1 - delta) / (1 + delta))

        # Error estimates (2% of values)
        err_betasw = betasw * 0.02
        err_beta90sw = beta90sw * 0.02
        err_bsw = bsw * 0.02

        return {
            'betasw': betasw,
            'beta90sw': beta90sw,
            'bsw': bsw,
            'err_betasw': err_betasw,
            'err_beta90sw': err_beta90sw,
            'err_bsw': err_bsw,
            'nsw': nsw
        }
        
    def refractive_index(std_RI):
        """
        Calculate the refractive index of seawater and its derivative with respect to salinity.

        This function computes the refractive index of seawater (`nsw`) and its derivative with respect to salinity (`dnswds`)
        using the Quan and Fry (1994) formulation, adjusted for the refractive index of air (Ciddor 1996).
        For wavelengths greater than or equal to 800 nm, the refractive index is adjusted to match a provided standard reference.

        Parameters
        ----------
        std_RI : array-like
            Standard reference refractive index values, aligned with the wavelength array `wl`.

        Returns
        -------
        dict
            Dictionary containing:
                'nsw' : array-like
                    Refractive index of seawater at each wavelength.
                'dnswds' : array-like
                    Derivative of the refractive index with respect to salinity at each wavelength.

        Notes
        -----
        - The function expects global variables `wl` (wavelength array), `S` (salinity), and `Tc` (temperature in degrees Celsius) to be defined.
        - For wavelengths >= 800 nm, the refractive index is adjusted to match the standard reference index (`std_RI`).
        """
        # std_RI is expected to be a numpy array or list aligned with wl wavelengths

        # Calculate refractive index of air (Ciddor 1996)
        n_air = 1.0 + (
            5792105.0 / (238.0185 - 1.0 / (wl / 1e3) ** 2) +
            167917.0 / (57.362 - 1.0 / (wl / 1e3) ** 2)
        ) / 1e8

        # Constants for seawater refractive index (Quan and Fry 1994)
        n0 = 1.31405
        n1 = 1.779e-4
        n2 = -1.05e-6
        n3 = 1.6e-8
        n4 = -2.02e-6
        n5 = 15.868
        n6 = 0.01155
        n7 = -0.00423
        n8 = -4382.0
        n9 = 1.1455e6

        # Calculate seawater refractive index
        nsw = (
            n0
            + (n1 + n2 * Tc + n3 * Tc**2) * S
            + n4 * Tc**2
            + (n5 + n6 * S + n7 * Tc) / wl
            + n8 / wl**2
            + n9 / wl**3
        )
        nsw *= n_air

        dnswds = (n1 + n2 * Tc + n3 * Tc**2 + n6 / wl) * n_air

        # Adjust nsw for wavelengths >= 800 nm using std_RI reference
        idx_800 = (wl == 800)
        offs = nsw[idx_800] - std_RI[idx_800]
        nsw[wl >= 800] = std_RI[wl >= 800] + offs

        return {'nsw': nsw, 'dnswds': dnswds}
        
    def BetaT():
        """
        Calculates the isothermal compressibility of seawater based on temperature and salinity.

        This function uses empirical formulas to estimate the secant bulk modulus of pure water and seawater,
        and then computes the isothermal compressibility.

        Returns:
            float: Isothermal compressibility of seawater (in Pa^-1).

        Notes:
            - The calculation is based on Millero (1980, Deep-sea Research) for pure water.
            - The function assumes the existence of global variables `Tc` (temperature in degrees Celsius)
              and `S` (salinity in PSU).
        """
        # pure water secant bulk Millero (1980, Deep-sea Research)
        kw = (19652.21 + 148.4206*Tc - 2.327105*Tc**2 +
              1.360477e-2*Tc**3 - 5.155288e-5*Tc**4)
        Btw_cal = 1 / kw
        # isothermal compressibility from Kell sound measurement in pure water
        # Btw = (50.88630+0.717582*Tc+0.7819867e-3*Tc^2+31.62214e-6*Tc^3-0.1323594e-6*Tc^4+0.634575e-9*Tc^5)/(1+21.65928e-3*Tc)*1e-6
        # seawater secant bulk        
        a0 = 54.6746 - 0.603459*Tc + 1.09987e-2*Tc**2 - 6.167e-5*Tc**3
        b0 = 7.944e-2 + 1.6483e-2*Tc - 5.3009e-4*Tc**2
        Ks = kw + a0*S + b0*S**1.5
        # calculate seawater isothermal compressibility from the secant bulk
        IsoComp = 1 / Ks * 1e-5  # unit in Pa
        return IsoComp
    
    def rhou_sw():
        """
        Calculate the density of seawater (kg/m^3) based on temperature and salinity.
        Uses the UNESCO 1981 (EOS-80) polynomial approximation for seawater density.
        Returns:
            float: Density of seawater in kg/m^3.
        Notes:
            - Requires global variables `Tc` (temperature in degrees Celsius) and `S` (salinity in PSU).
            - For pure water, set S = 0.
            - Reference: UNESCO, 1981. "Background papers and supporting data on the Practical Salinity Scale 1978." UNESCO Technical Papers in Marine Science, No. 37.
        """
        # density of water and seawater,unit is Kg/m^3, from UNESCO,38,1981
        a0, a1, a2, a3, a4 = 8.24493e-1, -4.0899e-3, 7.6438e-5, -8.2467e-7, 5.3875e-9
        a5, a6, a7, a8 = -5.72466e-3, 1.0227e-4, -1.6546e-6, 4.8314e-4
        b0, b1, b2, b3, b4, b5 = 999.842594, 6.793952e-2, -9.09529e-3, 1.001685e-4, -1.120083e-6, 6.536332e-9
        
        # density for pure water
        density_w = b0 + b1*Tc + b2*Tc**2 + b3*Tc**3 + b4*Tc**4 + b5*Tc**5
        
        # density for pure seawater
        density_sw = (density_w + 
                      (a0 + a1*Tc + a2*Tc**2 + a3*Tc**3 + a4*Tc**4)*S +
                      (a5 + a6*Tc + a7*Tc**2)*S**1.5 + a8*S**2)
        return density_sw
    
    def dlnasw_ds():
        """
        Calculates the partial derivative of the natural logarithm of water activity (ln(a_w)) 
        with respect to salinity (S) for seawater.

        This function uses a polynomial fit to data from Millero and Leung (1976, American Journal of Science, 276, 1035-1077),
        specifically Table 19, which was reproduced using Eqs. (14, 22, 23, 88, 107) and fitted to a polynomial equation.

        Returns:
            float: The value of d(ln(a_w))/dS, the partial derivative of the natural logarithm of water activity with respect to salinity.

        Notes:
            - The variables `Tc` (temperature in degrees Celsius) and `S` (salinity) must be defined in the scope where this function is called.
            - The coefficients are empirically derived for seawater.
        """
        # water activity data of seawater is from Millero and Leung (1976,American
        # Journal of Science,276,1035-1077). Table 19 was reproduced using
        # Eqs.(14,22,23,88,107) then were fitted to polynominal equation.
        # dlnawds is partial derivative of natural logarithm of water activity
        # w.r.t.salinity    
        dlnawds = ((-5.58651e-4 + 2.40452e-7*Tc - 3.12165e-9*Tc**2 + 2.40808e-11*Tc**3) +
               1.5*(1.79613e-5 - 9.9422e-8*Tc + 2.08919e-9*Tc**2 - 1.39872e-11*Tc**3)*S**0.5 +
               2*(-2.31065e-6 - 1.37674e-9*Tc - 1.93316e-11*Tc**2)*S)
        return dlnawds
    
    def PMH(n_wat):
        """
        Calculates the derivative of density with respect to refractive index for water.

        This function computes an expression involving the square of the refractive index (`n_wat`)
        and is typically used in optical property calculations.

        Parameters:
            n_wat (float): The refractive index of water.

        Returns:
            float: The calculated derivative of density with respect to the refractive index.
        """
        n_wat2 = n_wat**2
        n_density_derivative = (n_wat2 - 1) * (1 + 2.0/3.0 * (n_wat2 + 2.0) * (n_wat/3.0 - 1.0 / 3.0 / n_wat)**2)
        return n_density_derivative
    
    
    ########################### MAIN ###########################
    
    res_refri = read_refri_std()
    wl = res_refri['wl']
    std_RI = res_refri['refri']
            
    a = absorption()
        
    wl_abs = a['wl']
    b = scattering(wl_abs, std_RI=std_RI)
        
    water_abs = a['abso']
    water_sca = b['bsw']
    nw = b['nsw']
    
    water_abs_approx = interp1d(wl_abs, water_abs, kind='linear', bounds_error=False, fill_value='extrapolate')(wavelen)
    water_sca_approx = interp1d(wl_abs, water_sca, kind='linear', bounds_error=False, fill_value='extrapolate')(wavelen)
    nw_approx = interp1d(wl_abs, nw, kind='linear', bounds_error=False, fill_value='extrapolate')(wavelen)
    
    return {
        'wavelen': wavelen,
        'a': water_abs_approx,
        'b': water_sca_approx,
        'Temp': Tc,
        'Sal': S,
        'nw': nw_approx,
        'aw_version': aw_version
    }    
