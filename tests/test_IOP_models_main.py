import pytest
import numpy as np
import pandas as pd
from iopmodel.IOP_models_main import IOP_four_comp

def test_IOP_four_comp_default():
    """Test IOP_four_comp with default parameters."""
    result = IOP_four_comp()
    assert isinstance(result, dict)
    assert 'spec' in result
    assert 'parm' in result
    assert isinstance(result['spec'], pd.DataFrame)
    assert isinstance(result['parm'], pd.DataFrame)
    # Check that required columns exist in spec DataFrame
    expected_cols = [
        'wavelen', 'aph', 'ad', 'acdom', 'bph', 'bd', 'aw', 'bw',
        'ap', 'agp', 'bp', 'at', 'bt', 'bbph', 'bbd', 'bbp', 'bbw', 'bbt',
        'Rrs_L11', 'Rrs_AM03'
    ]
    for col in expected_cols:
        assert col in result['spec'].columns

def test_IOP_four_comp_custom_params():
    """Test IOP_four_comp with custom parameter values."""
    Chl = 2.5
    ag440 = 0.3
    ISM = 5
    Temp = 10
    Sal = 35
    S_cdom = 0.018
    bbdtilde = 0.03
    result = IOP_four_comp(
        Chl=Chl, ag440=ag440, ISM=ISM, Temp=Temp, Sal=Sal,
        S_cdom=S_cdom, bbdtilde=bbdtilde, aw_version=2, lib_cdom=False
    )
    parm = result['parm'].iloc[0]
    assert np.isclose(parm['Chl'], Chl)
    assert np.isclose(parm['ag440'], ag440)
    assert np.isclose(parm['ISM'], ISM)
    assert np.isclose(parm['Temp'], Temp)
    assert np.isclose(parm['Sal'], Sal)
    assert np.isclose(parm['S_cdom'], S_cdom)
    assert np.isclose(parm['bbdtilde'], bbdtilde)
    assert parm['aw_version'] == 2

def test_IOP_four_comp_frac_phyto():
    """Test IOP_four_comp with explicit frac_phyto."""
    frac_phyto = np.array([0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    result = IOP_four_comp(frac_phyto=frac_phyto)
    parm = result['parm'].iloc[0]
    assert np.allclose(parm['frac_phyto'], frac_phyto)

def test_IOP_four_comp_output_shapes():
    """Test that output DataFrames have matching lengths and expected shape."""
    result = IOP_four_comp()
    spec = result['spec']
    # All columns should have the same length
    lengths = [len(spec[col]) for col in spec.columns]
    assert len(set(lengths)) == 1
    # Should have more than 10 wavelengths
    assert len(spec) > 10

def test_IOP_four_comp_warn_neg_bd_flag():
    """Test that warn_neg_bd flag is present and is bool or None."""
    result = IOP_four_comp()
    warn_flag = result['warn_neg_bd']
    assert isinstance(warn_flag, (bool, type(None)))

def test_IOP_four_comp_attr_aph676():
    """Test that attr_aph676 is present and is float or None."""
    result = IOP_four_comp()
    attr = result['attr_aph676']
    assert isinstance(attr, (float, type(None)))

def test_IOP_four_comp_ag_seed_reproducibility():
    """Test that ag_seed gives reproducible results for acdom."""
    result1 = IOP_four_comp(ag_seed=42)
    result2 = IOP_four_comp(ag_seed=42)
    np.testing.assert_allclose(result1['spec']['acdom'], result2['spec']['acdom'])

def test_IOP_four_comp_lib_cdom_switch():
    """Test that lib_cdom switch changes acdom output."""
    result_lib = IOP_four_comp(lib_cdom=True)
    result_no_lib = IOP_four_comp(lib_cdom=False)
    # Should not be identical
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(result_lib['spec']['acdom'], result_no_lib['spec']['acdom'])