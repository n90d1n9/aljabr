package tech.kayys.aljabr.core.nn;

import tech.kayys.aljabr.core.tensor.Tensor;

/**
 * Fully connected (linear) layer: y = xW^T + b.
 */
public class Linear extends Module {

    private final long inFeatures;
    private final long outFeatures;
    private final boolean hasBias;

    public Linear(long inFeatures, long outFeatures, boolean bias) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.hasBias = bias;

        // Initialize weight (using randn)
        Tensor weight = Tensor.randn(outFeatures, inFeatures);
        registerParameter("weight", weight);

        if (bias) {
            Tensor biasParam = Tensor.zeros(outFeatures);
            registerParameter("bias", biasParam);
        }
    }

    public Linear(long inFeatures, long outFeatures) {
        this(inFeatures, outFeatures, true);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor weight = parameters.get("weight");
        Tensor output = input.matmul(weight.transpose(0, 1));

        if (hasBias) {
            Tensor bias = parameters.get("bias");
            output = output.add(bias);
        }
        return output;
    }

    public long inFeatures() {
        return inFeatures;
    }

    public long outFeatures() {
        return outFeatures;
    }

    @Override
    public String toString() {
        return String.format("Linear(in_features=%d, out_features=%d, bias=%b)", inFeatures, outFeatures, hasBias);
    }
}
