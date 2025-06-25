import numpy as np
import pandas as pd
from iopmodel.IOP_models_func import (
    IOP_d_B22_C2,
    IOP_ph_B22_C2,
    IOP_cdom_B22_C2,
    IOP_cdom_B22_C2_lib,
    rand_frac
)
from iopmodel.WOPP import WOPP
from iopmodel.misc import IOP2AOP_Lee_11, IOP2AOP_Albert_Mobley_03
from iopmodel.global_variables import wavelen_IOP, phyto_lib_info


def IOP_four_comp(
    Chl=1,
    ag440=0.18,
    ISM=1,
    Temp=20,
    Sal=15,
    qt_bd=0.5,
    qt_md=0.5,
    frac_phyto=None,
    S_cdom=0.017422,
    bbdtilde=0.0216,
    wavelen=wavelen_IOP,
    A_d=None,
    G_d=None,
    lib_cdom=True,
    aw_version=3,
    ag_seed=None,
    **kwargs
):
    """
    Compute inherent optical properties (IOPs) and related spectral quantities for a four-component water model.
    This function models the optical properties of water as a combination of four main components: phytoplankton, 
    detritus, colored dissolved organic matter (CDOM), and pure water. It calculates absorption and backscattering 
    spectra, as well as remote sensing reflectance estimates, based on input parameters describing the 
    concentrations and properties of these components.
    
    Parameters
    ----------
    Chl : float, optional
        Chlorophyll-a concentration (mg/m^3). Default is 1.
    ag440 : float, optional
        CDOM absorption at 440 nm (1/m). Default is 0.18.
    ISM : float, optional
        Inorganic suspended matter concentration (g/m^3). Default is 1.
    Temp : float, optional
        Water temperature (°C). Default is 20.
    Sal : float, optional
        Salinity (psu). Default is 15.
    qt_bd : float, optional
        Fraction of detrital backscattering. Default is 0.5.
    qt_md : float, optional
        Fraction of detrital mass-specific scattering. Default is 0.5.
    frac_phyto : array-like or None, optional
        Fractional composition of phytoplankton types. If None, random fractions are generated.
    S_cdom : float, optional
        CDOM spectral slope (nm^-1). Default is 0.017422.
    bbdtilde : float, optional
        Backscattering ratio for detritus. Default is 0.0216.
    wavelen : array-like, optional
        Wavelengths (nm) at which to compute spectra. Default is `wavelen_IOP`.
    A_d : array-like or None, optional
        Detrital absorption coefficients. Default is None.
    G_d : array-like or None, optional
        Detrital spectral shape parameters. Default is None.
    lib_cdom : bool, optional
        If True, use library-based CDOM model. Default is True.
    aw_version : int, optional
        Version of pure water absorption coefficients to use. Default is 3.
    ag_seed : int or None, optional
        Random seed for CDOM library selection. Default is None.
        Additional keyword arguments passed to phytoplankton IOP model.
        
    Returns
    -------
    result : dict
        Dictionary containing:
            - 'spec': pandas.DataFrame
                Spectral quantities (absorption, backscattering, reflectance, etc.) as columns.
            - 'parm': pandas.DataFrame
                Summary of input and derived parameters.
            - 'attr_aph676': float or None
                Attribute for aph(676) if available.
            - 'warn_neg_bd': bool or None
                Warning flag if negative detrital backscattering detected.
                
    Notes
    -----
    This function relies on several external helper functions and data structures, such as `IOP_d_B22_C2`, `IOP_ph_B22_C2`, `IOP_cdom_B22_C2_lib`, `WOPP`, and `phyto_lib_info`.
    """
    if frac_phyto is None:
        frac_phyto = np.array(list(rand_frac().values()))

    # Spectral components
    list_d = IOP_d_B22_C2(ISM, Chl, qt_bd=qt_bd, qt_md=qt_md,
                          wavelen=wavelen, A=A_d, G=G_d)
    list_ph = IOP_ph_B22_C2(Chl, frac_phyto, wavelen=wavelen, **kwargs)

    attr_aph676 = getattr(list_ph, "attr_aph676", None)

    if lib_cdom:
        list_cdom = IOP_cdom_B22_C2_lib(ag440, wavelen=wavelen, ag_seed=ag_seed)
    else:
        list_cdom = IOP_cdom_B22_C2(ag440, S_cdom, wavelen=wavelen)

    list_WOPP = WOPP(Temp, Sal, wavelen=wavelen, aw_version=aw_version)

    spec = {
        'wavelen': list_ph['wavelen'],
        'aph': list_ph['aph'],
        'ad': list_d['ad'],
        'acdom': list_cdom[0]['acdom'],
        'bph': list_ph['bph'],
        'bd': list_d['bd'],
        'aw': list_WOPP['a'],
        'bw': list_WOPP['b'],
    }

    spec['ap'] = spec['aph'] + spec['ad']
    spec['agp'] = spec['ap'] + spec['acdom']
    spec['bp'] = spec['bph'] + spec['bd']
    spec['at'] = spec['agp'] + spec['aw']
    spec['bt'] = spec['bp'] + spec['bw']

    # bbtilde blend
    bbtilde_phyto = phyto_lib_info['bbtilde']
    bbphtilde = np.sum(frac_phyto * bbtilde_phyto)

    spec['bbph'] = spec['bph'] * bbphtilde
    spec['bbd'] = spec['bd'] * bbdtilde
    spec['bbp'] = spec['bbph'] + spec['bbd']
    spec['bbw'] = spec['bw'] / 2.0
    spec['bbt'] = spec['bbp'] + spec['bbw']

    # Remote sensing reflectance estimates
    spec['Rrs_L11'] = IOP2AOP_Lee_11(spec['bbw'], spec['bbp'], spec['at'], spec['bbt'])
    spec['Rrs_AM03'] = IOP2AOP_Albert_Mobley_03(a=spec['at'], bb=spec['bbt'])

    # Parameters summary
    parm = {
        'Chl': Chl,
        'ag440': ag440,
        'ISM': ISM,
        'Temp': Temp,
        'Sal': Sal,
        'qt_bd': list_d['_parm']['qt_bd'],
        'qt_md': list_d['_parm']['qt_md'],
        'A_d': list_d['_parm']['A_cd'],
        'G_d': list_d['_parm']['G_cd'],
        'Albedo550_d': list_d['_parm']['Albedo_550'],
        'frac_phyto': frac_phyto,
        'ph_parm': list_ph.get('attr_ph_parm', None),
        'S_cdom': S_cdom,
        'bbdtilde': bbdtilde,
        'bbphtilde': bbphtilde,
        'aw_version': aw_version,
        'ag_seed': ag_seed
    }
    
    df_spec = pd.DataFrame(spec)
    df_parm = pd.DataFrame([parm])
    
    result = {
        'spec': df_spec,
        'parm': df_parm,
        'attr_aph676': attr_aph676,
        'warn_neg_bd': list_d.get('warn_neg_bd', None)
    }

    return result