import warnings

import numpy as np

from scripts.extract_uni2h_features import integral_image, rect_sum


def test_integral_image_rect_sum_uses_signed_accumulator_without_underflow():
    mask = np.ones((1100, 1100), dtype=bool)
    ii = integral_image(mask)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        total = rect_sum(ii, 900, 900, 1000, 1000)

    assert total == 100 * 100
    assert np.issubdtype(ii.dtype, np.signedinteger)
