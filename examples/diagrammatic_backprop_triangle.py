"""Tiny Diagrammatic Backprop triangle on a causal diagnosis task."""

import numpy as np

from catml import (
    DiagramMap,
    attention,
    collect_params,
    eval_map,
    linear,
    pullback_program,
    relu,
    sgd_step,
    sigmoid,
    trace_function,
    triangle_gradients,
)
from diagrammatic_backprop_dataset import (
    Sample,
    build_dataset,
    split_direct_supervision,
)


def tiny_transformer(
    x, hidden_dim: int, out_dim: int, rng: np.random.Generator, prefix: str
):
    """One attention block plus a tiny MLP head."""
    h = linear(x, hidden_dim, rng, tag=f"{prefix}_embed")
    h = relu(h + attention(h, h, h))
    ff = linear(h, 2 * hidden_dim, rng, tag=f"{prefix}_ff1")
    ff = relu(ff)
    ff = linear(ff, hidden_dim, rng, tag=f"{prefix}_ff2")
    h = relu(h + ff)
    return sigmoid(linear(h, out_dim, rng, tag=f"{prefix}_out"))


def build_maps(
    seed: int = 0, hidden_dim: int = 24
) -> tuple[DiagramMap, DiagramMap, DiagramMap]:
    """Build evidence->syndrome, syndrome->diagnosis, and evidence->diagnosis maps."""
    rng = np.random.default_rng(seed)
    samples = build_dataset()
    example_evidence = samples[0].evidence
    example_latent = samples[0].latent
    latent_dim = example_latent.shape[-1]
    diagnosis_dim = samples[0].diagnosis.shape[-1]

    f_program = trace_function(
        lambda x: tiny_transformer(x, hidden_dim, latent_dim, rng, "evidence_syndrome"),
        example_evidence,
    )
    g_program = trace_function(
        lambda x: tiny_transformer(
            x, hidden_dim, diagnosis_dim, rng, "syndrome_diagnosis"
        ),
        example_latent,
    )
    h_program = trace_function(
        lambda x: tiny_transformer(
            x, hidden_dim, diagnosis_dim, rng, "evidence_diagnosis"
        ),
        example_evidence,
    )
    return (
        DiagramMap("Evidence", "Syndrome", f_program, "f"),
        DiagramMap("Syndrome", "Diagnosis", g_program, "g"),
        DiagramMap("Evidence", "Diagnosis", h_program, "h"),
    )


def train(
    use_db: bool,
    seed: int = 0,
    epochs: int = 100,
    lr: float = 0.01,
    db_weight: float = 0.25,
):
    """Train with or without triangle consistency and report held-out accuracy."""
    old_err = np.seterr(over="ignore", divide="ignore", invalid="ignore")
    try:
        samples = build_dataset()
        f_map, g_map, h_map = build_maps(seed=seed)
        params = {
            "f": collect_params(f_map.program),
            "g": collect_params(g_map.program),
            "h": collect_params(h_map.program),
        }
        direct_train, held_out = split_direct_supervision(samples)

        for _ in range(epochs):
            for sample in samples:
                _supervised_step(f_map, params["f"], sample.evidence, sample.latent, lr)
                _clip_params(params["f"])
                _supervised_step(
                    g_map, params["g"], sample.latent, sample.diagnosis, lr
                )
                _clip_params(params["g"])

            for sample in direct_train:
                _supervised_step(
                    h_map, params["h"], sample.evidence, sample.diagnosis, lr
                )
                _clip_params(params["h"])

            if use_db:
                for sample in samples:
                    _, h_grads, f_grads, g_grads = triangle_gradients(
                        h_map,
                        f_map,
                        g_map,
                        sample.evidence,
                    )
                    scale = db_weight / sample.diagnosis.size
                    sgd_step(params["h"], _scale_grads(h_grads, scale), lr)
                    sgd_step(params["f"], _scale_grads(f_grads, scale), lr)
                    sgd_step(params["g"], _scale_grads(g_grads, scale), lr)
                    _clip_params(params["h"])
                    _clip_params(params["f"])
                    _clip_params(params["g"])

        direct_accuracy = _accuracy(h_map, direct_train)
        held_out_accuracy = _accuracy(h_map, held_out)
        triangle_error = np.mean(
            [
                np.mean(
                    (
                        eval_map(h_map, sample.evidence)
                        - eval_map(g_map, eval_map(f_map, sample.evidence))
                    )
                    ** 2
                )
                for sample in held_out
            ]
        )
        return {
            "direct_accuracy": round(direct_accuracy, 4),
            "held_out_accuracy": round(held_out_accuracy, 4),
            "triangle_error": round(float(triangle_error), 4),
            "held_out_count": len(held_out),
        }
    finally:
        np.seterr(**old_err)


def main():
    baseline = train(use_db=False)
    db = train(use_db=True)
    print("Direct supervision only:", baseline)
    print("With diagrammatic backprop:", db)


def _supervised_step(
    map_: DiagramMap,
    params_by_tag: dict[str, dict[str, np.ndarray]],
    x: np.ndarray,
    target: np.ndarray,
    lr: float,
) -> float:
    pred = eval_map(map_, x)
    residual = pred - target
    _, _, grads = pullback_program(map_.program, (x,), residual / residual.size)
    sgd_step(params_by_tag, grads, lr)
    return float(0.5 * np.mean(residual * residual))


def _accuracy(map_: DiagramMap, samples: list[Sample]) -> float:
    correct = 0
    for sample in samples:
        pred = eval_map(map_, sample.evidence)
        if int(np.argmax(pred, axis=-1)[0]) == int(
            np.argmax(sample.diagnosis, axis=-1)[0]
        ):
            correct += 1
    return correct / max(1, len(samples))


def _scale_grads(
    grads: dict[str, dict[str, np.ndarray]], scale: float
) -> dict[str, dict[str, np.ndarray]]:
    return {
        tag: {key: scale * value for key, value in tag_grads.items()}
        for tag, tag_grads in grads.items()
    }


def _clip_params(
    params_by_tag: dict[str, dict[str, np.ndarray]], limit: float = 3.0
) -> None:
    for params in params_by_tag.values():
        for key, value in params.items():
            params[key] = np.clip(value, -limit, limit)


if __name__ == "__main__":
    main()
