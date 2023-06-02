import torch
from fast_transformers.builders import TransformerEncoderBuilder

# Create the builder for our transformers
builder = TransformerEncoderBuilder.from_kwargs(
    n_layers=8,
    n_heads=8,
    query_dimensions=64,
    value_dimensions=64,
    feed_forward_dimensions=1024
)

# Build a transformer with softmax attention
builder.attention_type = "full"
softmax_model = builder.get()

# Build a transformer with linear attention
builder.attention_type = "linear"
linear_model = builder.get()

if __name__ == '__main__':
    # Construct the dummy input
    X = torch.rand(10, 2000, 8*64)

    # Prepare everythin for CUDA
    X = X.cuda()
    softmax_model.cuda()
    softmax_model.eval()
    linear_model.cuda()
    linear_model.eval()

    # Warmup the GPU
    # with torch.no_grad():
    #     softmax_model(X)
    #     linear_model(X)
    # torch.cuda.synchronize()

    # Measure the execution time
    softmax_start = torch.cuda.Event(enable_timing=True)
    softmax_end = torch.cuda.Event(enable_timing=True)
    linear_start = torch.cuda.Event(enable_timing=True)
    linear_end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        softmax_start.record()
        y = softmax_model(X)
        softmax_end.record()
        torch.cuda.synchronize()
        print("Softmax: ", softmax_start.elapsed_time(softmax_end), "ms")
        # Softmax: 144 ms (on a GTX1080Ti)

    with torch.no_grad():
        linear_start.record()
        y = linear_model(X)
        linear_end.record()
        torch.cuda.synchronize()
        print("Linear: ", linear_start.elapsed_time(linear_end), "ms")
        # Linear: 68 ms (on a GTX1080Ti)