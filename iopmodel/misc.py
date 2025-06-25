import numpy as np

def rnorm_bound(n, mean, sd=None, lo=-np.inf, up=np.inf, cv=None, **kwargs):
    """
    Generate random samples from a normal distribution with optional bounds.
    Parameters:
        n (int): Number of samples to generate.
        mean (float): Mean of the normal distribution.
        sd (float, optional): Standard deviation of the normal distribution. Required if cv is not provided.
        lo (float, optional): Lower bound for the samples. Defaults to -np.inf (no lower bound).
        up (float, optional): Upper bound for the samples. Defaults to np.inf (no upper bound).
        cv (float, optional): Coefficient of variation. If provided, sd is calculated as abs(cv * mean).
        **kwargs: Additional keyword arguments (currently unused).
        
    Returns:
        np.ndarray: Array of n samples drawn from the normal distribution, constrained between lo and up.
        
    Raises:
        ValueError: If neither sd nor cv is provided.
        
    Notes:
        - This is probably very much not optimal and could be done in a much more reasonable way. However,
          this project is a direct port of the IOPModel from R to python and that is how it was in R.
    """
    if cv is not None:
        sd = abs(cv * mean)
    if sd is None:
        raise ValueError("At least need sd or cv")

    x = np.random.normal(loc=mean, scale=sd, size=n)
    x = x[(x >= lo) & (x <= up)]
    
    while len(x) < n:
        needed = n - len(x)
        x_new = np.random.normal(loc=mean, scale=sd, size=needed)
        x_new = x_new[(x_new >= lo) & (x_new <= up)]
        x = np.concatenate([x, x_new])
    
    return x


def vec_to_mat(x, n, axis=1):
    """
    Converts a 1D vector into a 2D matrix by repeating it along the specified axis.

    Parameters
    ----------
    x : array_like
        Input 1D vector to be repeated.
    n : int
        Number of repetitions along the specified axis.
    axis : int, optional
        Axis along which to repeat the vector:
        - 1: Repeat as rows (default).
        - 2: Repeat as columns.

    Returns
    -------
    result : ndarray
        2D matrix with the vector repeated along the specified axis.

    Raises
    ------
    ValueError
        If `axis` is not 1 or 2.

    Examples
    --------
    >>> vec_to_mat([1, 2, 3], 2, axis=1)
    array([[1, 2, 3],
           [1, 2, 3]])

    >>> vec_to_mat([1, 2, 3], 2, axis=2)
    array([[1, 1],
           [2, 2],
           [3, 3]])
    """
    x = np.asarray(x)
    if axis == 1:
        result = np.tile(x, (n, 1))
    elif axis == 2:
        result = np.tile(x.reshape(-1, 1), (1, n))
    else:
        raise ValueError("`axis` must be 1 (rows) or 2 (columns)")
    return result


def def_frac(name, value, name_phyto):
    """
    Creates a numpy array of zeros with the same length as `name_phyto`, and sets the element corresponding to `name` to `value`.

    Parameters:
        name (str): The name to search for in `name_phyto`.
        value (float): The value to assign at the index corresponding to `name`.
        name_phyto (list of str): List of names to use as reference for indexing.

    Returns:
        numpy.ndarray: Array of zeros with `value` set at the index of `name`.

    Raises:
        ValueError: If `name` is not found in `name_phyto`.
    """
    v = np.zeros(len(name_phyto))
    name_index = {n: i for i, n in enumerate(name_phyto)}
    if name not in name_index:
        raise ValueError(f"{name} not found in name_phyto")
    v[name_index[name]] = value
    return v


def IOP2AOP_Albert_Mobley_03(a, bb, theta_s=30, theta_v=0, windspd=7):
    """
    Estimates remote sensing reflectance (Rrs) from inherent optical properties (IOPs) using the Albert & Mobley (2003) model.
    
    Parameters
    ----------
    a : float or np.ndarray
        Total absorption coefficient (1/m).
    bb : float or np.ndarray
        Total backscattering coefficient (1/m).
    theta_s : float, optional
        Solar zenith angle in degrees (default is 30).
    theta_v : float, optional
        Sensor/viewing zenith angle in degrees (default is 0).
    windspd : float, optional
        Wind speed at the surface in m/s (default is 7).
        
    Returns
    -------
    Rrs : float or np.ndarray
        Remote sensing reflectance (sr^-1) estimated from the input IOPs.
        
    References
    ----------
    Albert, A. & Mobley, C.D. (2003). An analytical model for subsurface irradiance and remote sensing reflectance in deep and shallow case-2 waters. Optics Express, 11(22), 2873-2890.
    """
    x = bb / (bb + a)
    
    p1 = 0.0512
    p2 = 4.6659
    p3 = -7.8387
    p4 = 5.4571
    p5 = 0.1098
    p6 = -0.0044
    p7 = 0.4021
    
    rrs = (p1 * x * (1 + p2 * x + p3 * x**2 + p4 * x**3) *
           (1 + p5 / np.cos(np.radians(theta_s))) *
           (1 + p7 / np.cos(np.radians(theta_v))) *
           (1 + p6 * windspd))
    
    Rrs = 0.526 * rrs / (1 - 2.164 * rrs)
    return Rrs


def IOP2AOP_Lee_11(bbw, bbp, at, bbt,
                   gw0=0.05881474, gw1=0.05062697,
                   gp0=0.03997009, gp1=0.1398902):
    """
    Computes the remote sensing reflectance (rrs) using the Lee et al. (2011) IOP to AOP model.
    
    Parameters:
        bbw (float): Backscattering coefficient of water.
        bbp (float): Backscattering coefficient of particles.
        at (float): Total absorption coefficient.
        bbt (float): Total backscattering coefficient.
        gw0 (float, optional): Empirical coefficient for water (default: 0.05881474).
        gw1 (float, optional): Empirical coefficient for water (default: 0.05062697).
        gp0 (float, optional): Empirical coefficient for particles (default: 0.03997009).
        gp1 (float, optional): Empirical coefficient for particles (default: 0.1398902).
        
    Returns:
        float: Remote sensing reflectance (rrs) calculated using the Lee et al. (2011) model.
        
    References:
        Lee, Z., et al. (2011). "An empirical formula to estimate remote sensing reflectance from inherent optical properties of natural waters." Journal of Geophysical Research: Oceans, 116(C2).
    """
    
    k = at + bbt
    
    rrs = (gw0 + gw1 * (bbw / k)) * (bbw / k) + \
          (gp0 + gp1 * (bbp / k)) * (bbp / k)
    
    return rrs