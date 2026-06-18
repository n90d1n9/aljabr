package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Sigmoid activation layer.
 */
public class Sigmoid extends Module {

    @Override
    public Tensor forward(Tensor input) {
        return input.sigmoid();
    }

    @Override
    public String toString() {
        return "Sigmoid()";
    }
}
