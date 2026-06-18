package tech.kayys.aljabr.backend.cpu;

import tech.kayys.aljabr.core.backend.ComputeBackend;
import tech.kayys.aljabr.core.tensor.*;
import tech.kayys.aljabr.core.memory.*;
import java.lang.foreign.ValueLayout;
import java.lang.foreign.MemorySegment;
import java.util.List;

public final class CpuBackend implements ComputeBackend {

    private static final String TODO = "CPU kernel not yet implemented";

    @Override
    public Tensor add(Tensor a, Tensor b) {
        Shape shape = a.shape();
        long n = shape.numel();
        CpuBuffer outBuf = new CpuBuffer(n * 4);
        var sa = ((DefaultTensor) a).buffer().segment();
        var sb = ((DefaultTensor) b).buffer().segment();
        var so = outBuf.segment();
        for (long i = 0; i < n; i++) {
            float va = sa.get(ValueLayout.JAVA_FLOAT, i * 4);
            float vb = sb.get(ValueLayout.JAVA_FLOAT, i * 4);
            so.set(ValueLayout.JAVA_FLOAT, i * 4, va + vb);
        }
        return new DefaultTensor(shape, DType.F32, DeviceType.CPU, outBuf, this);
    }

    @Override
    public Tensor sub(Tensor a, Tensor b) {
        Shape shape = a.shape();
        long n = shape.numel();
        CpuBuffer outBuf = new CpuBuffer(n * 4);
        var sa = ((DefaultTensor) a).buffer().segment();
        var sb = ((DefaultTensor) b).buffer().segment();
        var so = outBuf.segment();
        for (long i = 0; i < n; i++) {
            float va = sa.get(ValueLayout.JAVA_FLOAT, i * 4);
            float vb = sb.get(ValueLayout.JAVA_FLOAT, i * 4);
            so.set(ValueLayout.JAVA_FLOAT, i * 4, va - vb);
        }
        return new DefaultTensor(shape, DType.F32, DeviceType.CPU, outBuf, this);
    }

    @Override
    public Tensor mul(Tensor a, float scalar) {
        Shape shape = a.shape();
        long n = shape.numel();
        CpuBuffer outBuf = new CpuBuffer(n * 4);
        var sa = ((DefaultTensor) a).buffer().segment();
        var so = outBuf.segment();
        for (long i = 0; i < n; i++) {
            float va = sa.get(ValueLayout.JAVA_FLOAT, i * 4);
            so.set(ValueLayout.JAVA_FLOAT, i * 4, va * scalar);
        }
        return new DefaultTensor(shape, DType.F32, DeviceType.CPU, outBuf, this);
    }

    @Override
    public Tensor mul(Tensor a, Tensor b) {
        Shape shape = a.shape();
        long n = shape.numel();
        CpuBuffer outBuf = new CpuBuffer(n * 4);
        var sa = ((DefaultTensor) a).buffer().segment();
        var sb = ((DefaultTensor) b).buffer().segment();
        var so = outBuf.segment();
        for (long i = 0; i < n; i++) {
            so.set(ValueLayout.JAVA_FLOAT, i * 4,
                    sa.get(ValueLayout.JAVA_FLOAT, i * 4) *
                    sb.get(ValueLayout.JAVA_FLOAT, i * 4));
        }
        return new DefaultTensor(shape, DType.F32, DeviceType.CPU, outBuf, this);
    }

    @Override
    public Tensor div(Tensor a, float scalar) {
        return mul(a, 1.0f / scalar);
    }

    @Override
    public Tensor div(Tensor a, Tensor b) {
        Shape shape = a.shape();
        long n = shape.numel();
        CpuBuffer outBuf = new CpuBuffer(n * 4);
        var sa = ((DefaultTensor) a).buffer().segment();
        var sb = ((DefaultTensor) b).buffer().segment();
        var so = outBuf.segment();
        for (long i = 0; i < n; i++) {
            so.set(ValueLayout.JAVA_FLOAT, i * 4,
                    sa.get(ValueLayout.JAVA_FLOAT, i * 4) /
                    sb.get(ValueLayout.JAVA_FLOAT, i * 4));
        }
        return new DefaultTensor(shape, DType.F32, DeviceType.CPU, outBuf, this);
    }

    @Override
    public Tensor addScalar(Tensor a, float scalar) {
        Shape shape = a.shape();
        long n = shape.numel();
        CpuBuffer outBuf = new CpuBuffer(n * 4);
        var sa = ((DefaultTensor) a).buffer().segment();
        var so = outBuf.segment();
        for (long i = 0; i < n; i++) {
            so.set(ValueLayout.JAVA_FLOAT, i * 4,
                    sa.get(ValueLayout.JAVA_FLOAT, i * 4) + scalar);
        }
        return new DefaultTensor(shape, DType.F32, DeviceType.CPU, outBuf, this);
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        throw new UnsupportedOperationException("matmul not yet implemented in CpuBackend");
    }

    @Override
    public Tensor reshape(Tensor a, long... newShape) {
        return new DefaultTensor(new Shape(newShape), a.dtype(), a.device(),
                ((DefaultTensor) a).buffer(), this);
    }

    @Override
    public Tensor attention(Tensor Q, Tensor K, Tensor V) {
        try {
            return FlashAttentionCpu.forward(Q, K, V,
                    Runtime.getRuntime().availableProcessors());
        } catch (Exception e) {
            return naiveAttention(Q, K, V);
        }
    }

    @Override
    public Tensor softmax(Tensor a) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor slice(Tensor a, long[] offsets, long[] sizes) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public List<Tensor> split(Tensor a, int axis, int parts) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor pow(Tensor a, float exponent) {
        Shape shape = a.shape();
        long n = shape.numel();
        CpuBuffer outBuf = new CpuBuffer(n * 4);
        var sa = ((DefaultTensor) a).buffer().segment();
        var so = outBuf.segment();
        for (long i = 0; i < n; i++) {
            so.set(ValueLayout.JAVA_FLOAT, i * 4,
                    (float) Math.pow(sa.get(ValueLayout.JAVA_FLOAT, i * 4), exponent));
        }
        return new DefaultTensor(shape, DType.F32, DeviceType.CPU, outBuf, this);
    }

    @Override
    public Tensor mean(Tensor a) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor abs(Tensor a) {
        Shape shape = a.shape();
        long n = shape.numel();
        CpuBuffer outBuf = new CpuBuffer(n * 4);
        var sa = ((DefaultTensor) a).buffer().segment();
        var so = outBuf.segment();
        for (long i = 0; i < n; i++) {
            so.set(ValueLayout.JAVA_FLOAT, i * 4,
                    Math.abs(sa.get(ValueLayout.JAVA_FLOAT, i * 4)));
        }
        return new DefaultTensor(shape, DType.F32, DeviceType.CPU, outBuf, this);
    }

    @Override
    public Tensor crossEntropy(Tensor pred, Tensor target) { throw new UnsupportedOperationException(TODO); }

    @Override
    public Tensor binaryCrossEntropy(Tensor pred, Tensor target) { throw new UnsupportedOperationException(TODO); }

    @Override
    public Tensor cast(Tensor a, tech.kayys.aljabr.core.tensor.DType dtype) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor to(Tensor a, tech.kayys.aljabr.core.tensor.DeviceType device) {
        throw new UnsupportedOperationException(TODO);
    }

    @Override
    public Tensor zerosLike(Tensor a) {
        Shape shape = a.shape();
        long n = shape.numel();
        CpuBuffer outBuf = new CpuBuffer(n * 4);
        return new DefaultTensor(shape, DType.F32, DeviceType.CPU, outBuf, this);
    }

    @Override
    public Tensor sqrt(Tensor a) {
        Shape shape = a.shape();
        long n = shape.numel();
        CpuBuffer outBuf = new CpuBuffer(n * 4);
        var sa = ((DefaultTensor) a).buffer().segment();
        var so = outBuf.segment();
        for (long i = 0; i < n; i++) {
            so.set(ValueLayout.JAVA_FLOAT, i * 4,
                    (float) Math.sqrt(sa.get(ValueLayout.JAVA_FLOAT, i * 4)));
        }
        return new DefaultTensor(shape, DType.F32, DeviceType.CPU, outBuf, this);
    }

    @Override public Tensor relu(Tensor a)     { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor sigmoid(Tensor a)  { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor tanh(Tensor a)     { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor log(Tensor a)      { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor exp(Tensor a)      { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor silu(Tensor a)     { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor flatten(Tensor a)  { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor unsqueeze(Tensor a, int dim) { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor squeeze(Tensor a)  { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor transpose(Tensor a){ throw new UnsupportedOperationException(TODO); }
    @Override public Tensor transpose(Tensor a, int d0, int d1) { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor gelu(Tensor a) { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor softmax(Tensor a, int dim) { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor logSoftmax(Tensor a, int dim) { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor mean(Tensor a, int dim, boolean keepDim) { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor sum(Tensor a) { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor sum(Tensor a, int dim, boolean keepDim) { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor max(Tensor a) { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor layerNorm(Tensor input, long[] normalizedShape, Tensor weight, Tensor bias, float eps) { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor rmsNorm(Tensor input, Tensor weight, float eps) { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor batchNorm(Tensor input, Tensor weight, Tensor bias, Tensor runningMean, Tensor runningVar, boolean training, float momentum, float eps) { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor conv2d(Tensor input, Tensor weight, Tensor bias, int stride, int padding, int dilation, int groups) { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor maxPool2d(Tensor input, int kernelSize, int stride, int padding) { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor adaptiveAvgPool2d(Tensor input, int outputH, int outputW) { throw new UnsupportedOperationException(TODO); }
    @Override public Tensor dropout(Tensor input, float p, boolean training) { throw new UnsupportedOperationException(TODO); }

    @Override
    public Tensor embedding(Tensor weight, Tensor input, long paddingIdx) {
        long[] inputDims = input.shape().dims();
        long vocabSize = weight.shape().dim(0);
        long embeddingDim = weight.shape().dim(1);

        long[] outputDims = new long[inputDims.length + 1];
        System.arraycopy(inputDims, 0, outputDims, 0, inputDims.length);
        outputDims[inputDims.length] = embeddingDim;

        long numElements = input.numel();
        CpuBuffer outBuf = new CpuBuffer(numElements * embeddingDim * 4);
        var sw = ((DefaultTensor) weight).buffer().segment();
        var si = ((DefaultTensor) input).buffer().segment();
        var so = outBuf.segment();

        DType inputDType = input.dtype();

        for (long i = 0; i < numElements; i++) {
            long idx;
            if (inputDType == DType.I32) {
                idx = si.get(ValueLayout.JAVA_INT, i * 4);
            } else if (inputDType == DType.I8) {
                idx = si.get(ValueLayout.JAVA_BYTE, i);
            } else {
                idx = (long) si.get(ValueLayout.JAVA_FLOAT, i * 4);
            }

            long outOffset = i * embeddingDim * 4;
            if (idx == paddingIdx || idx < 0 || idx >= vocabSize) {
                for (long j = 0; j < embeddingDim; j++) {
                    so.set(ValueLayout.JAVA_FLOAT, outOffset + j * 4, 0f);
                }
            } else {
                long weightOffset = idx * embeddingDim * 4;
                MemorySegment.copy(sw, weightOffset, so, outOffset, embeddingDim * 4);
            }
        }

        return new DefaultTensor(new Shape(outputDims), DType.F32, DeviceType.CPU, outBuf, this);
    }

    @Override
    public long numel(Tensor a) {
        return a.numel();
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private Tensor naiveAttention(Tensor Q, Tensor K, Tensor V) {
        Tensor scores = matmul(Q, transpose(K));
        float scale = (float) (1.0 / Math.sqrt(Q.shape().dim(1)));
        scores = mul(scores, scale);
        Tensor probs = softmax(scores);
        return matmul(probs, V);
    }
}