import numpy as np

from catml import eval_program, grad_program, trace_function
from catml import linear, relu, sigmoid


def _collect_params(program):
    params_by_tag = {}
    for instr in program.instructions:
        if instr.tag is None or not instr.params:
            continue
        params_by_tag[instr.tag] = instr.params
    return params_by_tag


def _sgd_step(params_by_tag, grads_by_tag, lr):
    for tag, params in params_by_tag.items():
        grads = grads_by_tag.get(tag)
        if grads is None:
            continue
        for key, value in grads.items():
            params[key] = params[key] - lr * value


def main():
    rng = np.random.default_rng(0)
    hidden_dim = 4

    def model(x):
        y = linear(x, hidden_dim, rng, tag="lin1")
        y = relu(y)
        y = linear(y, 1, rng, tag="lin2")
        return sigmoid(y)

    x0 = np.array([0.0, 0.0])
    program = trace_function(model, x0)
    params_by_tag = _collect_params(program)

    # Dataset
    data_x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    data_y = np.array([[0.0], [1.0], [1.0], [0.0]])

    lr = 1e-1
    epochs = 1000
    for epoch in range(epochs):
        total_loss = 0.0
        for x, target in zip(data_x, data_y):
            pred = eval_program(program, (x,))
            err = pred - target
            loss = float(0.5 * np.sum(err * err))
            total_loss += loss
            out_grad = err

            # Compute grads and take an SGD step.
            _, grads = grad_program(program, (x,), out_grad)
            _sgd_step(params_by_tag, grads, lr)
        if epoch == 0: print({k: {p: g.shape for p, g in v.items()} for k, v in grads.items()})
        if epoch % 100 == 0:
            print(f"epoch {epoch} loss {total_loss:.4f}")

    for x, target in zip(data_x, data_y):
        pred = eval_program(program, (x,))
        print("x", x, "pred", pred, "target", target)


if __name__ == "__main__":
    main()
