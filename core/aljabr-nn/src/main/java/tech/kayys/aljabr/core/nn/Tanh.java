package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Tanh activation layer.
 */
public class Tanh extends Module {

    @Override
    public Tensor forward(Tensor input) {
        return input.tanh();
    }

    @Override
    public String toString() {
        return "Tanh()";
    }
}
