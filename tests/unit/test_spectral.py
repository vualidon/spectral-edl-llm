from __future__ import annotations

import numpy as np


def test_knn_graph_is_symmetrized():
    from skedl.core.graph import symmetrized_knn_graph

    kernel = np.array(
        [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ]
    )

    graph = symmetrized_knn_graph(kernel, k=1, include_self=False)

    assert graph.shape == kernel.shape
    assert np.allclose(graph, graph.T)
    assert np.all(np.diag(graph) == 0.0)


def test_lambda2_risk_increases_for_weaker_connectivity():
    from skedl.core.graph import symmetrized_knn_graph
    from skedl.core.spectral import connectivity_risk, lambda2_normalized_laplacian

    strong_kernel = np.array(
        [
            [1.0, 0.8, 0.75, 0.7],
            [0.8, 1.0, 0.72, 0.71],
            [0.75, 0.72, 1.0, 0.78],
            [0.7, 0.71, 0.78, 1.0],
        ]
    )
    weak_kernel = np.array(
        [
            [1.0, 0.9, 0.05, 0.02],
            [0.9, 1.0, 0.04, 0.01],
            [0.05, 0.04, 1.0, 0.88],
            [0.02, 0.01, 0.88, 1.0],
        ]
    )

    strong_graph = symmetrized_knn_graph(strong_kernel, k=2, include_self=False)
    weak_graph = symmetrized_knn_graph(weak_kernel, k=2, include_self=False)

    strong_lambda2 = lambda2_normalized_laplacian(strong_graph)
    weak_lambda2 = lambda2_normalized_laplacian(weak_graph)

    strong_risk = connectivity_risk(strong_lambda2, tau=5.0)
    weak_risk = connectivity_risk(weak_lambda2, tau=5.0)

    assert strong_lambda2 > weak_lambda2
    assert weak_risk > strong_risk
