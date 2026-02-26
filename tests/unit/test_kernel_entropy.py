from __future__ import annotations

import numpy as np


def test_cosine_kernel_is_symmetric_and_bounded():
    from skedl.core.kernel import cosine_kernel

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    kernel = cosine_kernel(embeddings)

    assert kernel.shape == (3, 3)
    assert np.allclose(kernel, kernel.T)
    assert np.all(kernel >= 0.0)
    assert np.all(kernel <= 1.0)
    assert np.allclose(np.diag(kernel), 1.0)


def test_trace_normalized_kernel_entropy_is_lower_for_identical_embeddings():
    from skedl.core.entropy import kernel_von_neumann_entropy
    from skedl.core.kernel import cosine_kernel

    identical = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=float)
    spread = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)

    h_identical = kernel_von_neumann_entropy(cosine_kernel(identical))
    h_spread = kernel_von_neumann_entropy(cosine_kernel(spread))

    assert h_identical < h_spread
    assert h_identical >= 0.0
