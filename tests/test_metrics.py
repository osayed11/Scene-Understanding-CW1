import numpy as np
from scipy.spatial.transform import Rotation as R
from metrics import evaluate
from trajectory import umeyama


def synthetic(n=200):
    t = np.arange(n) * 0.05
    xyz = np.stack([np.sin(t), np.cos(t), 0.1 * t], 1)
    q = R.from_euler("z", t).as_quat()
    return np.column_stack([t, xyz, q])


def test_umeyama_recovers_similarity():
    gt = synthetic()
    s, Rm, tr = 2.5, R.from_euler("xyz", [0.3, -0.2, 0.9]).as_matrix(), np.array([1, 2, 3.0])
    src = ((gt[:, 1:4] - tr) @ Rm) / s          # inverse transform
    s2, R2, t2 = umeyama(src, gt[:, 1:4], with_scale=True)
    assert abs(s2 - s) < 1e-6 and np.allclose(R2, Rm) and np.allclose(t2, tr)


def test_evaluate_identical_is_zero():
    gt = synthetic()
    m = evaluate(gt, gt, rpe_delta=5)
    assert m.ate_noscale < 1e-9 and m.ate_scale < 1e-9 and abs(m.scale - 1) < 1e-9
    assert m.rpe_trans < 1e-9 and m.tracked == 1.0 and m.lost_spans == []


def test_scaled_estimate_has_scale_ate_zero():
    gt = synthetic()
    est = gt.copy(); est[:, 1:4] *= 0.4   # monocular-style scale error
    m = evaluate(est, gt, rpe_delta=5)
    assert m.ate_scale < 1e-9 and abs(m.scale - 2.5) < 1e-6 and m.ate_noscale > 0.1


def test_lost_span_detected():
    gt = synthetic()
    est = np.delete(gt, slice(80, 120), axis=0)
    m = evaluate(est, gt, rpe_delta=5)
    assert len(m.lost_spans) == 1 and m.tracked < 1.0
