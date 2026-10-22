import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from quantize_custom import (
    ACTIVATION_BITS,
    BIAS_BITS,
    WEIGHT_BITS,
    SCALE_SHIFT,
    QuantizedConvRelu2d,
    QuantizedLinear,
    QuantizationMarker,
    QuantizeActivation,
    dequantize,
    export_mem,
    fuse_conv_relu,
    insert_quantization_markers,
    quantize,
    quantize_calibrated,
    quantize_layer_parameters,
    quantize_weights,
    signed_quantization_range,
)
import torch
from model import SimpleCNN


ACTIVATION_MIN = -(2 ** (ACTIVATION_BITS - 1))
ACTIVATION_MAX = 2 ** (ACTIVATION_BITS - 1) - 1
WEIGHT_MIN = -(2 ** (WEIGHT_BITS - 1))
WEIGHT_MAX = 2 ** (WEIGHT_BITS - 1) - 1
BIAS_MIN = -(2 ** (BIAS_BITS - 1))
BIAS_MAX = 2 ** (BIAS_BITS - 1) - 1


def random_uniform(low, high, size=(), *, generator, dtype=torch.float64):
    return torch.empty(size, dtype=dtype).uniform_(low, high, generator=generator)


class QuantizeTestCase(unittest.TestCase):
    def test_signed_ranges(self):
        for bits, expected in ((1, (-1, 0)), (4, (-8, 7)), (8, (-128, 127))):
            with self.subTest(bits=bits):
                self.assertEqual(signed_quantization_range(bits), expected)

    def test_rounds_and_clips_to_logical_width_not_storage_width(self):
        values = torch.tensor([-100.0, -4.25, -0.25, 0.25, 0.75, 100.0])
        scale = torch.tensor(0.5)
        zero = torch.tensor(1)
        for dtype in (torch.int8, torch.int16, torch.int32):
            with self.subTest(dtype=dtype):
                actual = quantize(values, scale, zero, bits=4, dtype=dtype)
                expected = torch.tensor([-8, -8, 0, 2, 2, 7], dtype=dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_wide_clipping_bounds_are_exact_for_float32_inputs(self):
        for bits in (BIAS_BITS, 32):
            with self.subTest(bits=bits):
                lower = -(2 ** (bits - 1))
                upper = 2 ** (bits - 1) - 1
                values = torch.tensor([lower * 2, upper * 2], dtype=torch.float32)
                actual = quantize(
                    values, torch.tensor(1.0), torch.tensor(0), bits, torch.int32
                )
                torch.testing.assert_close(
                    actual,
                    torch.tensor([lower, upper], dtype=torch.int32),
                    rtol=0,
                    atol=0,
                )

    def test_rejects_invalid_widths_and_storage_types(self):
        for bits, dtype in (
            (0, torch.int8),
            (-1, torch.int8),
            (9, torch.int8),
            (33, torch.int32),
            (8, torch.float32),
            (8, torch.uint8),
        ):
            with self.subTest(bits=bits, dtype=dtype):
                with self.assertRaises(ValueError):
                    quantize(
                        torch.tensor(1.0),
                        torch.tensor(1.0),
                        torch.tensor(0),
                        bits,
                        dtype,
                    )


class QuantizeLayerParametersTestCase(unittest.TestCase):
    def test_bias_uses_configured_width_with_int32_storage(self):
        weights = torch.zeros((3, 2))
        bias = torch.tensor([-(2.0**BIAS_BITS), 0.0, 2.0**BIAS_BITS])
        _, int_bias, _ = quantize_layer_parameters(weights, bias, torch.tensor(1.0))
        torch.testing.assert_close(
            int_bias,
            torch.tensor([BIAS_MIN, 0, BIAS_MAX], dtype=torch.int32),
            rtol=0,
            atol=0,
        )

    def test_absent_bias_uses_int32_zeros(self):
        _, int_bias, _ = quantize_layer_parameters(
            torch.ones((3, 2)), None, torch.tensor(1.0)
        )
        torch.testing.assert_close(
            int_bias, torch.zeros(3, dtype=torch.int32), rtol=0, atol=0
        )

    def test_quantizes_each_output_channel_with_its_own_scale(self):
        weights = torch.tensor(
            [[1.0, -1.0], [10.0, -10.0], [0.0, 0.0]], dtype=torch.float32
        )
        output_channel_axis = 0

        int_weights, scales = quantize_weights(weights, axis=output_channel_axis)

        self.assertEqual(scales.shape, (3,))
        self.assertTrue(torch.all(scales > 0).item())
        torch.testing.assert_close(int_weights[0], int_weights[1], rtol=0, atol=0)
        torch.testing.assert_close(
            int_weights[2], torch.zeros(2, dtype=torch.int8), rtol=0, atol=0
        )


class QuantizedConvolutionTestCase(unittest.TestCase):
    def test_rejects_even_kernel_dimensions(self):
        bias = torch.tensor([0], dtype=torch.int32)
        w_scale = torch.tensor([0.5])
        x_scale = torch.tensor(1.0)
        x_zero = torch.tensor(0, dtype=torch.int8)
        y_scale = torch.tensor(1.0)
        y_zero = torch.tensor(0, dtype=torch.int8)

        for kernel_h, kernel_w in ((2, 3), (3, 2)):
            with self.subTest(kernel_h=kernel_h, kernel_w=kernel_w):
                weights = torch.ones((1, 1, kernel_h, kernel_w), dtype=torch.int8)
                with self.assertRaises(AssertionError):
                    QuantizedConvRelu2d(
                        weights=weights,
                        bias=bias,
                        w_scale=w_scale,
                        x_scale=x_scale,
                        x_zero=x_zero,
                        y_scale=y_scale,
                        y_zero=y_zero,
                    )

    def test_requantization_multiply_does_not_overflow_int32(self):
        weights = torch.tensor([[[[1]]]], dtype=torch.int8)
        bias = torch.tensor([0], dtype=torch.int32)
        w_scale = torch.tensor([0.5])
        x_scale = torch.tensor(1.0)
        x_zero = torch.tensor(0, dtype=torch.int8)
        y_scale = torch.tensor(1.0)
        y_zero = torch.tensor(0, dtype=torch.int8)

        layer = QuantizedConvRelu2d(
            weights=weights,
            bias=bias,
            w_scale=w_scale,
            x_scale=x_scale,
            x_zero=x_zero,
            y_scale=y_scale,
            y_zero=y_zero,
        )
        image = torch.tensor([[[[4]]]], dtype=torch.int8)

        actual = layer(image)
        expected = torch.tensor([[[[2]]]], dtype=torch.int8)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_zero_input_gets_zero_point(self):
        kernel = torch.tensor(
            [
                [-1, 2, -3],
                [4, -5, 6],
                [7, -8, 9],
            ],
            dtype=torch.int8,
        )

        x_zero = torch.tensor(3, dtype=torch.int8)
        y_zero = torch.tensor(40, dtype=torch.int8)

        layer = QuantizedConvRelu2d(
            weights=kernel.reshape(1, 1, 3, 3),
            bias=torch.tensor([0], dtype=torch.int32),
            w_scale=torch.tensor([0.7]),
            x_scale=torch.tensor(0.5),
            x_zero=x_zero,
            y_scale=torch.tensor(2.0),
            y_zero=y_zero,
        )

        image = torch.full((1, 1, 4, 4), fill_value=x_zero)

        result = layer.forward(image)

        torch.testing.assert_close(
            result,
            torch.full((1, 1, 4, 4), fill_value=y_zero, dtype=result.dtype),
            rtol=0,
            atol=0,
        )

    def test_zero_weight_gets_zero_point(self):
        kernel = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=torch.int8,
        )

        y_zero = torch.tensor(40, dtype=torch.int8)

        layer = QuantizedConvRelu2d(
            weights=kernel.reshape(1, 1, 3, 3),
            bias=torch.tensor([0], dtype=torch.int32),
            w_scale=torch.tensor([0.7]),
            x_scale=torch.tensor(0.5),
            x_zero=torch.tensor(3, dtype=torch.int8),
            y_scale=torch.tensor(2.0),
            y_zero=y_zero,
        )

        image = torch.full((1, 1, 4, 4), fill_value=torch.tensor(10, dtype=torch.int8))

        result = layer.forward(image)

        torch.testing.assert_close(
            result,
            torch.full((1, 1, 4, 4), fill_value=y_zero, dtype=result.dtype),
            rtol=0,
            atol=0,
        )

    def test_batched_same_as_single(self):
        generator = torch.Generator().manual_seed(10)
        NUM_IMAGES = 20
        IN_CHANNELS = 3
        OUT_CHANNELS = 2
        KERNEL_HEIGHT = 5
        KERNEL_WIDTH = 5
        HEIGHT = 30
        WIDTH = 50

        weights = torch.randint(
            WEIGHT_MIN,
            WEIGHT_MAX,
            (OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH),
            dtype=torch.int8,
            generator=generator,
        )

        layer = QuantizedConvRelu2d(
            weights=weights,
            bias=torch.randint(
                BIAS_MIN,
                BIAS_MAX + 1,
                (OUT_CHANNELS,),
                dtype=torch.int32,
                generator=generator,
            ),
            w_scale=random_uniform(
                0, 2, (OUT_CHANNELS,), generator=generator, dtype=torch.float32
            ),
            x_scale=random_uniform(0, 2, generator=generator, dtype=torch.float32),
            x_zero=torch.randint(
                ACTIVATION_MIN,
                ACTIVATION_MAX,
                tuple(),
                dtype=torch.int8,
                generator=generator,
            ),
            y_scale=random_uniform(0, 2, generator=generator, dtype=torch.float32),
            y_zero=torch.randint(
                ACTIVATION_MIN,
                ACTIVATION_MAX,
                tuple(),
                dtype=torch.int8,
                generator=generator,
            ),
        )

        images = torch.randint(
            ACTIVATION_MIN,
            ACTIVATION_MAX,
            (NUM_IMAGES, IN_CHANNELS, HEIGHT, WIDTH),
            dtype=torch.int8,
            generator=generator,
        )

        individual_results = torch.stack(
            [layer.forward(image.unsqueeze(0)).squeeze() for image in images]
        )
        batched_results = layer.forward(images)

        torch.testing.assert_close(individual_results, batched_results, rtol=0, atol=0)

    def test_matches_requantized_float_convolution(self):
        NUM_CASES = 100
        FIRST_SEED = 1000
        NUM_IMAGES = 2
        IN_CHANNELS = 2
        OUT_CHANNELS = 3
        KERNEL_HEIGHT = 3
        KERNEL_WIDTH = 3
        HEIGHT = 5
        WIDTH = 7
        BIAS_RANGE = 8000

        for case in range(NUM_CASES):
            seed = FIRST_SEED + case
            with self.subTest(seed=seed):
                generator = torch.Generator().manual_seed(seed)

                # Sample scales logarithmically to cover several orders of magnitude.
                x_scale = 10 ** random_uniform(-3, 0, generator=generator)
                w_scale = 10 ** random_uniform(
                    -4, -1, (OUT_CHANNELS,), generator=generator
                )
                max_multiplier = 10 ** random_uniform(-5.05, -0.05, generator=generator)

                # Keep the effective scale below 1 for the fixed-point multiplier.
                y_scale = x_scale * w_scale.max() / max_multiplier
                x_zero = torch.randint(
                    ACTIVATION_MIN,
                    ACTIVATION_MAX + 1,
                    (),
                    generator=generator,
                    dtype=torch.int8,
                )
                y_zero = torch.randint(
                    ACTIVATION_MIN,
                    ACTIVATION_MAX + 1,
                    (),
                    generator=generator,
                    dtype=torch.int8,
                )

                images_q = torch.randint(
                    ACTIVATION_MIN,
                    ACTIVATION_MAX + 1,
                    (NUM_IMAGES, IN_CHANNELS, HEIGHT, WIDTH),
                    generator=generator,
                    dtype=torch.int8,
                )
                images = dequantize(images_q, x_scale, x_zero)
                weights_q = torch.randint(
                    WEIGHT_MIN,
                    WEIGHT_MAX + 1,
                    (OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH),
                    generator=generator,
                    dtype=torch.int8,
                )
                weights = dequantize(
                    weights_q,
                    w_scale.reshape(OUT_CHANNELS, 1, 1, 1),
                    torch.tensor(0, dtype=torch.int8),
                )
                bias_q = torch.randint(
                    -BIAS_RANGE // 2,
                    BIAS_RANGE // 2,
                    (OUT_CHANNELS,),
                    generator=generator,
                    dtype=torch.int32,
                )
                bias = dequantize(
                    bias_q, x_scale * w_scale, torch.tensor(0, dtype=torch.int32)
                )

                self._check_float_reference(
                    images, weights, bias, x_scale, x_zero, w_scale, y_scale, y_zero
                )

    def test_zero_point_extremes_match_float_convolution(self):
        # Opposite weights exercise both ReLU and saturation at either input zero point.
        images = torch.tensor([[[[-255.0, 0.0, 255.0]]]], dtype=torch.float64)
        weights = torch.tensor([2.0, -2.0], dtype=torch.float64).reshape(2, 1, 1, 1)
        bias = torch.zeros(2, dtype=torch.float64)
        x_scale = torch.tensor(1.0, dtype=torch.float64)
        w_scale = torch.tensor([0.25, 0.25], dtype=torch.float64)
        y_scale = torch.tensor(1.0, dtype=torch.float64)

        for input_zero in (ACTIVATION_MIN, ACTIVATION_MAX):
            for output_zero in (ACTIVATION_MIN, ACTIVATION_MAX):
                with self.subTest(input_zero=input_zero, output_zero=output_zero):
                    convolved, expected = self._check_float_reference(
                        images,
                        weights,
                        bias,
                        x_scale,
                        torch.tensor(input_zero, dtype=torch.int8),
                        w_scale,
                        y_scale,
                        torch.tensor(output_zero, dtype=torch.int8),
                    )
                    self.assertTrue(torch.any(convolved < 0).item())
                    self.assertTrue(torch.any(convolved > 0).item())
                    self.assertTrue(torch.any(expected == ACTIVATION_MAX).item())

    def _check_float_reference(
        self, images, weights, bias, x_scale, x_zero, w_scale, y_scale, y_zero
    ):
        # Reference method: https://leimao.github.io/article/Neural-Networks-Quantization/
        # Compute with dequantized operands, rather than the original floats,
        # so input/weight/bias quantization error is excluded from the comparison.
        weight_scales = w_scale.reshape(-1, 1, 1, 1)
        bias_scales = x_scale * w_scale
        w_zero = torch.tensor(0, dtype=torch.int8)
        bias_zero = torch.tensor(0, dtype=torch.int32)
        images_q = quantize(images, x_scale, x_zero, ACTIVATION_BITS, torch.int8)
        weights_q = quantize(weights, weight_scales, w_zero, WEIGHT_BITS, torch.int8)
        bias_q = quantize(bias, bias_scales, bias_zero, BIAS_BITS, torch.int32)

        images_dq = dequantize(images_q, x_scale, x_zero)
        weights_dq = dequantize(weights_q, weight_scales, w_zero)
        bias_dq = dequantize(bias_q, bias_scales, bias_zero)

        convolved = torch.nn.functional.conv2d(
            images_dq,
            weights_dq,
            bias_dq,
            padding=(weights.size(2) // 2, weights.size(3) // 2),
        )
        float_result = torch.relu(convolved)
        expected = quantize(float_result, y_scale, y_zero, ACTIVATION_BITS, torch.int8)

        layer = QuantizedConvRelu2d(
            weights=weights_q,
            bias=bias_q,
            w_scale=w_scale,
            x_scale=x_scale,
            x_zero=x_zero,
            y_scale=y_scale,
            y_zero=y_zero,
        )
        actual = layer(images_q)

        self.assertEqual(actual.dtype, torch.int8)
        self.assertEqual(actual.shape, expected.shape)

        # The blog post suggests checking for exact equality here
        # We don't do this because we actually introduce approximation
        # error when we do our shift trick, which breaks exactness.
        torch.testing.assert_close(
            actual.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1
        )

        # Check ReLU gives 0 exactly
        self.assertTrue(torch.all(actual >= y_zero).item())
        torch.testing.assert_close(
            actual[convolved <= 0], expected[convolved <= 0], rtol=0, atol=0
        )

        return convolved, expected


# ngl chat wrote these tests so I haven't looked at them too much
# i mostly wrote the ones above though
class QuantizedLinearTestCase(unittest.TestCase):
    def test_zero_input_gets_quantized_bias(self):
        weights = torch.tensor([[2, -3, 5], [-7, 11, 13]], dtype=torch.int8)
        x_zero = torch.tensor(-17, dtype=torch.int8)
        y_zero = torch.tensor(23, dtype=torch.int8)
        bias = torch.tensor([8, -12], dtype=torch.int32)
        w_scale = torch.tensor([0.25, 0.125])
        x_scale = torch.tensor(0.5)
        y_scale = torch.tensor(1.0)
        bias_scale = x_scale * w_scale
        bias_zero = torch.tensor(0, dtype=torch.int32)

        layer = QuantizedLinear(
            weights=weights,
            bias=bias,
            w_scale=w_scale,
            x_scale=x_scale,
            x_zero=x_zero,
            y_scale=y_scale,
            y_zero=y_zero,
        )
        self.assertEqual(layer.full_bias.dtype, torch.int32)

        inputs = torch.full((4, 3), x_zero, dtype=torch.int8)
        actual = layer(inputs)
        expected = quantize(
            dequantize(bias, bias_scale, bias_zero),
            y_scale,
            y_zero,
            ACTIVATION_BITS,
            torch.int8,
        ).expand(4, -1)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_supports_vector_and_leading_batch_dimensions(self):
        layer = QuantizedLinear(
            weights=torch.tensor([[1, 2, 3], [-4, 5, -6]], dtype=torch.int8),
            bias=torch.tensor([10, -20], dtype=torch.int32),
            w_scale=torch.tensor([0.25, 0.125]),
            x_scale=torch.tensor(0.5),
            x_zero=torch.tensor(3, dtype=torch.int8),
            y_scale=torch.tensor(1.0),
            y_zero=torch.tensor(-4, dtype=torch.int8),
        )
        inputs = torch.tensor(
            [[[3, 3, 3], [4, 5, 6]], [[-2, 7, 9], [10, -11, 12]]],
            dtype=torch.int8,
        )

        batched = layer(inputs)
        self.assertEqual(batched.shape, (2, 2, 2))
        individual = torch.stack(
            [torch.stack([layer(vector) for vector in batch]) for batch in inputs]
        )
        torch.testing.assert_close(batched, individual, rtol=0, atol=0)

    def test_output_zero_point_is_added_before_saturation(self):
        layer = QuantizedLinear(
            weights=torch.tensor([[127]], dtype=torch.int8),
            bias=torch.tensor([0], dtype=torch.int32),
            w_scale=torch.tensor([0.5]),
            x_scale=torch.tensor(1.0),
            x_zero=torch.tensor(0, dtype=torch.int8),
            y_scale=torch.tensor(1.0),
            y_zero=torch.tensor(100, dtype=torch.int8),
        )

        inputs = torch.tensor([[127], [-128]], dtype=torch.int8)
        actual = layer(inputs)
        expected = torch.tensor([[ACTIVATION_MAX], [ACTIVATION_MIN]], dtype=torch.int8)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_matches_requantized_float_linear(self):
        generator = torch.Generator().manual_seed(2087)

        for case in range(100):
            with self.subTest(case=case):
                batch_shape = (2, 3)
                in_features = 7
                out_features = 5
                x_scale = 10 ** random_uniform(-3, 0, generator=generator)
                w_scale = 10 ** random_uniform(
                    -4, -1, (out_features,), generator=generator
                )
                max_multiplier = 10 ** random_uniform(-5.05, -0.05, generator=generator)
                y_scale = x_scale * w_scale.max() / max_multiplier
                x_zero = torch.randint(
                    ACTIVATION_MIN,
                    ACTIVATION_MAX + 1,
                    (),
                    generator=generator,
                    dtype=torch.int8,
                )
                y_zero = torch.randint(
                    ACTIVATION_MIN,
                    ACTIVATION_MAX + 1,
                    (),
                    generator=generator,
                    dtype=torch.int8,
                )
                inputs_q = torch.randint(
                    ACTIVATION_MIN,
                    ACTIVATION_MAX + 1,
                    (*batch_shape, in_features),
                    generator=generator,
                    dtype=torch.int8,
                )
                weights_q = torch.randint(
                    WEIGHT_MIN,
                    WEIGHT_MAX + 1,
                    (out_features, in_features),
                    generator=generator,
                    dtype=torch.int8,
                )
                bias_q = torch.randint(
                    -8000, 8001, (out_features,), generator=generator, dtype=torch.int32
                )
                weight_zero = torch.tensor(0, dtype=torch.int8)
                bias_zero = torch.tensor(0, dtype=torch.int32)

                inputs_dq = dequantize(inputs_q, x_scale, x_zero)
                weights_dq = dequantize(
                    weights_q,
                    w_scale.reshape(-1, 1),
                    weight_zero,
                )
                bias_scale = x_scale * w_scale
                bias_dq = dequantize(bias_q, bias_scale, bias_zero)
                expected = quantize(
                    torch.nn.functional.linear(inputs_dq, weights_dq, bias_dq),
                    y_scale,
                    y_zero,
                    ACTIVATION_BITS,
                    torch.int8,
                )

                layer = QuantizedLinear(
                    weights=weights_q,
                    bias=bias_q,
                    w_scale=w_scale,
                    x_scale=x_scale,
                    x_zero=x_zero,
                    y_scale=y_scale,
                    y_zero=y_zero,
                )
                actual = layer(inputs_q)

                self.assertEqual(actual.dtype, torch.int8)
                self.assertEqual(actual.shape, expected.shape)
                torch.testing.assert_close(
                    actual.to(torch.int32), expected.to(torch.int32), rtol=0, atol=1
                )


class QuantizeCalibratedTestCase(unittest.TestCase):
    def test_converts_calibrated_graph_to_integer_modules(self):
        generator = torch.Generator().manual_seed(490)
        model = SimpleCNN().eval()
        traced = torch.fx.symbolic_trace(model)
        marked = insert_quantization_markers(fuse_conv_relu(traced))
        calibration_images = torch.rand((2, 1, 28, 28), generator=generator)
        marked(calibration_images)

        quantized = quantize_calibrated(marked)
        result = quantized(calibration_images)
        module_types = {type(module) for module in quantized.modules()}

        self.assertEqual(result.shape, (2, 10))
        self.assertEqual(result.dtype, torch.int8)
        self.assertIn(QuantizeActivation, module_types)
        self.assertIn(QuantizedConvRelu2d, module_types)
        self.assertIn(QuantizedLinear, module_types)
        self.assertNotIn(QuantizationMarker, module_types)


class ExportMemTestCase(unittest.TestCase):
    def test_exports_kernels_biases_and_qparams_in_graph_order(self):
        root = torch.nn.Module()

        # Register linear first to ensure export follows execution, not registration.
        root.linear = QuantizedLinear(
            weights=torch.tensor([[-128, 127], [1, -1]], dtype=torch.int8),
            bias=torch.tensor([-(2**31), 2**31 - 1], dtype=torch.int32),
            w_scale=torch.tensor([0.25, 0.125]),
            x_scale=torch.tensor(1.0),
            x_zero=torch.tensor(0, dtype=torch.int8),
            y_scale=torch.tensor(1.0),
            y_zero=torch.tensor(-128, dtype=torch.int8),
        )
        root.conv = QuantizedConvRelu2d(
            weights=torch.tensor([[[[1]], [[-2]]], [[[3]], [[4]]]], dtype=torch.int8),
            bias=torch.tensor([-300, 400], dtype=torch.int32),
            w_scale=torch.tensor([0.5, 0.25]),
            x_scale=torch.tensor(1.0),
            x_zero=torch.tensor(-3, dtype=torch.int8),
            y_scale=torch.tensor(1.0),
            y_zero=torch.tensor(127, dtype=torch.int8),
        )
        root.input_quant = QuantizeActivation(
            torch.tensor(0.5), torch.tensor(-3, dtype=torch.int8)
        )

        graph = torch.fx.Graph()
        value = graph.placeholder("x")
        for name in ("input_quant", "conv", "linear"):
            value = graph.call_module(name, (value,))
        graph.output(value)
        model = torch.fx.GraphModule(root, graph)

        with TemporaryDirectory() as directory:
            output = Path(directory) / "weights.mem"
            export_mem(model, output)
            lines = output.read_text().splitlines()
            metadata = json.loads(output.with_suffix(".json").read_text())

        self.assertTrue(all(len(line) == 2 for line in lines))

        data = bytes(int(line, 16) for line in lines)
        self.assertEqual(metadata["scale_approx_bits"], SCALE_SHIFT)
        self.assertEqual(
            metadata["input_quantization"],
            [{"name": "input_quant", "scale": 0.5, "zero": -3}],
        )

        layer_names = [layer["name"] for layer in metadata["layers"]]
        self.assertEqual(layer_names, ["conv", "linear"])

        def check_tensor(descriptor, expected, offset):
            self.assertEqual(descriptor["offset"], offset)
            self.assertEqual(descriptor["shape"], list(expected.shape))

            width = expected.element_size()
            length = descriptor["length"]
            actual = [
                int.from_bytes(data[index : index + width], "little", signed=True)
                for index in range(offset, offset + length, width)
            ]
            self.assertEqual(actual, expected.reshape(-1).tolist())

            return offset + length

        offset = 0

        for layer in metadata["layers"]:
            module = model.get_submodule(layer["name"])
            self.assertEqual(layer["offset"], offset)
            self.assertEqual(layer["weight_shape"], list(module.weights.shape))

            for channel, kernel in enumerate(layer["kernels"]):
                offset = check_tensor(
                    kernel["weights"], module.weights[channel], offset
                )
                offset = check_tensor(
                    kernel["full_bias"], module.full_bias[channel], offset
                )

            for name, descriptor in layer["quantization"].items():
                offset = check_tensor(descriptor, getattr(module, name), offset)

            self.assertEqual(layer["length"], offset - layer["offset"])

        self.assertEqual(offset, len(data))
