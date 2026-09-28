import numpy as np
from amaf.analysis.metrics import normalized_auc
from amaf.analysis.statistics import confidence_interval

def test_auc_constant():
    assert abs(normalized_auc(np.array([0,1,2.]),np.array([3,3,3.]))-3.0)<1e-9

def test_ci_singleton():
    assert confidence_interval([2.0])==(2.0,2.0)
