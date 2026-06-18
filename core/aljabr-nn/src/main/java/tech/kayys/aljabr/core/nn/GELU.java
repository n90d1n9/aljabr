package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * GELU activation layer.
 */
public class GELU extends Module {

    @Override
    public Tensor forward(Tensor input) {
        return input.gelu();
    }

    @Override
    public String toString() {
        return "GELU()";
    }
}
