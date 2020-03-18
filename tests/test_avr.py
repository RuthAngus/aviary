import numpy as np
import aviary as ay
import matplotlib.pyplot as plt
import pandas as pd

def test_avr():
    p_mp, p_mr, p_a = ay.get_avr_coefficients()

    age_mr = ay.v_to_age(10, p_mr)
    age_mp = ay.v_to_age(10, p_mp)
    age_a = ay.v_to_age(10, p_a)

    v_mr = ay.age_to_v(age_mr, p_mr)
    v_mp = ay.age_to_v(age_mp, p_mp)
    v_a = ay.age_to_v(age_a, p_a)

    assert np.isclose(v_mr, 10, atol=1e-6)
    assert np.isclose(v_mp, 10, atol=1e-6)
    assert np.isclose(v_a, 10, atol=1e-6)


if __name__ == "__main__":
    test_avr()
