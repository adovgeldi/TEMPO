import pytest
from tempo_forecasting.models import KNNModel
import numpy as np


n = 5
a = np.arange(4*n).reshape((4,n))
b = np.array([np.arange(2*n)[::2],np.arange(3*n)[::3],np.arange(4*n)[::4],np.arange(5*n)[::5]])


@pytest.mark.parametrize("arr, steps, expected_output", [
    (np.arange(5), 
     3, 
     [[0., 1., 2.],[1., 2., 3.],[2., 3., 4.]]),
])
def test_build_subsequences(arr, steps, expected_output):
    model = KNNModel(target_y="Demand", date_col="Date")
    output = model._build_subsequences(arr, steps)
    assert np.all(output == expected_output)


@pytest.mark.parametrize("arr, expected_output", [
    (a, [[2.],[2.],[2.],[2.]]),
    (b, [[4.],[6.],[8.],[10.]])
])
def test_calculate_complexity_estimates(arr, expected_output):
    model = KNNModel(target_y="Demand", date_col="Date")
    output = model._calculate_complexity_estimates(arr)
    assert np.all(output == expected_output)


@pytest.mark.parametrize("a, b, expected_rounded_output", [
    (a,
     b,
     [[2.],[3.],[4.],[5.]]),
])
def test_calculate_complexity_correction_factors(a,b,expected_rounded_output):
    model = KNNModel(target_y="Demand", date_col="Date")
    ce_a = model._calculate_complexity_estimates(a)
    ce_b = model._calculate_complexity_estimates(b)
    ccf = model._calculate_complexity_correction_factors(ce_a,ce_b)
    rounded_ccf = np.round(ccf,2) # rounding because the epsilons throw it off a tiny bit
    assert np.all(rounded_ccf == expected_rounded_output)


@pytest.mark.parametrize("a, b, expected_output", [
    (a[:,:4],
     a[:,:4]+1,
     [[2.],[2.],[2.],[2.]]),
])
def test_calculate_euclidean_distances(a,b,expected_output):
    model = KNNModel(target_y="Demand", date_col="Date")
    output = model._calculate_euclidean_distances(a,b)
    assert np.all(output == expected_output)


@pytest.mark.parametrize("q, s, expected_rounded_output", [
    (np.array([a[0]]),
     b,
     [[ 10.95],[ 32.86],[ 65.73],[109.54]]),
])
def test_calculate_complexity_invariant_distances(q,s,expected_rounded_output):
    model = KNNModel(target_y="Demand", date_col="Date")
    cid = model._calculate_complexity_invariant_distances(standardized_q = q,
                                                        standardized_s = s)
    rounded_cid = np.round(cid,2)
    assert np.all(rounded_cid == expected_rounded_output)

test_calculate_complexity_invariant_distances(np.array([a[0]]),b,[[ 10.95],[ 32.86],[ 65.73],[109.54]])