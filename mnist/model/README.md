# CNN Accelerator Model Pipeline

This folder mostly contains the model training and quantization pipeline for the
CNN accelerator (specifically the MNIST network in this case).

## Quantized Weight Export (.mem)

From `mnist/model`, run:

```sh
uv run python quantize_custom.py --export-format mem --output weights.mem
```

`--export-format torch` (the default) saves the PyTorch state dictionary instead.
Without `--output`, the filename is `mnist_cnn_int8.pt` or `mnist_cnn_int8.mem`.

The memory export produces `weights.mem` and `weights.json`:

- `.mem` contains one two-digit hex byte per line, suitable for `$readmemh`
  into an 8-bit memory. Signed integers use two's complement; int32 values
  occupy four little-endian bytes, with no alignment padding.
- Layers appear in FX graph execution order. For each output channel, all its
  weights are followed by its
  int32 `full_bias`. This bias already includes the input-zero-point correction.
- Convolution weights use `[output_channel, input_channel, height, width]`
  order, with width changing fastest. Linear weights use
  `[output_feature, input_feature]` order, with input feature changing fastest.
- After all channels come the int32 `approx_m` multiplier(s), int8 `x_zero`
  (convolutions only), and int8 `y_zero`. Multipliers use the
  file-level `scale_approx_bits` fractional bits.
- The JSON sidecar gives byte offsets, lengths, and shapes for every
  kernel, bias, and quantization tensor, plus layer names/types and the input
  quantizer's scale and zero point. Offsets are zero-based and equal `.mem`
  line numbers minus one.

### Winograd export

`export_winograd_mem(quantized_model, Path("weights.mem"))` writes the same
layout as `export_mem`, except each 3x3 convolution kernel is replaced by its Winograd
F(2x2, 3x3) tile `U' = G' g G'^T` with `G' = 2G`:

- Each kernel is `[input_channel, 4, 4]` int16 values (two little-endian bytes,
  row-major), in place of the `[input_channel, 3, 3]` int8 weights. Values fit
  in 12 signed bits, matching the `u_weights` port of `winograd_3x3_conv.sv`.
- Biases, `approx_m`, zero points and linear layers are unchanged.
- The layer's JSON has `"weight_format": "winograd_f2x2_3x3"` and a `winograd`
  entry with `u_shape`, `u_bits` and `output_shift`. Direct layers have
  `"weight_format": "direct"`.
- With `G' = 2G` the Winograd result is 4x the direct convolution. Shift it
  right by `output_shift` (2) before adding `full_bias`. Sum `U' (.) V` over
  input channels first, then apply the output transform once.
