import pytest
import numpy as np

from iopmodel.misc import (
    rnorm_bound,
    vec_to_mat,
    def_frac,
    IOP2AOP_Albert_Mobley_03,
    IOP2AOP_Lee_11
)

def test_rnorm_bound_basic():
    np.random.seed(0)
    samples = rnorm_bound(1000, mean=5, sd=2, lo=3, up=7)
    assert np.all(samples >= 3)
    assert np.all(samples <= 7)
    assert len(samples) == 1000
    # Test with cv instead of sd
    samples_cv = rnorm_bound(100, mean=10, cv=0.1, lo=8, up=12)
    assert np.all(samples_cv >= 8)
    assert np.all(samples_cv <= 12)
    assert len(samples_cv) == 100

def test_rnorm_bound_raises():
    with pytest.raises(ValueError):
        rnorm_bound(10, mean=0)

def test_vec_to_mat_axis1():
    x = [1, 2, 3]
    mat = vec_to_mat(x, 2, axis=1)
    expected = np.array([[1, 2, 3], [1, 2, 3]])
    np.testing.assert_array_equal(mat, expected)

def test_vec_to_mat_axis2():
    x = [1, 2, 3]
    mat = vec_to_mat(x, 2, axis=2)
    expected = np.array([[1, 1], [2, 2], [3, 3]])
    np.testing.assert_array_equal(mat, expected)

def test_vec_to_mat_invalid_axis():
    with pytest.raises(ValueError):
        vec_to_mat([1, 2, 3], 2, axis=0)

def test_def_frac_basic():
    names = ['a', 'b', 'c']
    v = def_frac('b', 5, names)
    expected = np.array([0, 5, 0])
    np.testing.assert_array_equal(v, expected)

def test_def_frac_not_found():
    names = ['a', 'b', 'c']
    with pytest.raises(ValueError):
        def_frac('d', 1, names)

def test_IOP2AOP_Albert_Mobley_03_scalar():
    a = 0.5
    bb = 0.2
    result = IOP2AOP_Albert_Mobley_03(a, bb)
    assert isinstance(result, float) or isinstance(result, np.floating)
    assert result > 0

def test_IOP2AOP_Albert_Mobley_03_array():
    a = np.array([0.5, 0.6])
    bb = np.array([0.2, 0.3])
    result = IOP2AOP_Albert_Mobley_03(a, bb)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)

def test_IOP2AOP_Lee_11_basic():
    rrs = IOP2AOP_Lee_11(bbw=0.01, bbp=0.02, at=0.1, bbt=0.03)
    assert isinstance(rrs, float) or isinstance(rrs, np.floating)
    assert rrs > 0

def test_IOP2AOP_Lee_11_array():
    bbw = np.array([0.01, 0.02])
    bbp = np.array([0.02, 0.03])
    at = np.array([0.1, 0.2])
    bbt = np.array([0.03, 0.04])
    rrs = IOP2AOP_Lee_11(bbw, bbp, at, bbt)
    assert isinstance(rrs, np.ndarray)
    assert rrs.shape == (2,)