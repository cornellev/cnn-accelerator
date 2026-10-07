import argparse
from copy import deepcopy
import time
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, fx, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleCNN


def make_loader(data_dir: Path, batch_size: int, train: bool) -> DataLoader:
    dataset = datasets.MNIST(
        data_dir,
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.inference_mode()
def calibrate(model: nn.Module, loader: DataLoader, batches: int) -> None:
    """Run representative images through observers to determine activation ranges."""
    for batch_index, (images, _) in enumerate(loader):
        if batch_index >= batches:
            break
        model(images)


@torch.inference_mode()
def accuracy(model: nn.Module, loader: DataLoader) -> float:
    correct = 0
    total = 0
    for images, labels in loader:
        correct += (model(images).argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return correct / total


class ConvRelu2d(nn.Module):
    """'Fused' 2D convolution + ReLU. Basically a marker for quantization."""

    def __init__(self, conv: nn.Conv2d, relu: nn.ReLU) -> None:
        super().__init__()
        self.conv = conv
        self.relu = relu

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(inputs))


def fuse_conv_relu(m: fx.GraphModule) -> fx.GraphModule:
    """Fuse Conv/ReLU pairs in an independent copy of the input graph module."""
    gm = deepcopy(m)
    modules = dict(gm.named_modules())

    for conv_node in list(gm.graph.nodes):
        conv_node: fx.Node

        if conv_node.op != "call_module":
            continue

        conv = modules[cast(str, conv_node.target)]
        if type(conv) is not nn.Conv2d:
            continue

        if len(conv_node.users) != 1:
            continue

        relu_node = next(iter(conv_node.users.keys()))
        if relu_node.op != "call_module":
            continue
        relu = modules[cast(str, relu_node.target)]
        if type(relu) is not nn.ReLU:
            continue

        with gm.graph.inserting_before(conv_node):
            fused_name = f"fused_{conv_node.name}_{relu_node.name}"
            gm.add_module(fused_name, ConvRelu2d(conv, relu))
            fused_node = gm.graph.call_module(
                fused_name, args=conv_node.args, kwargs=conv_node.kwargs
            )

        relu_node.replace_all_uses_with(fused_node)
        gm.graph.erase_node(relu_node)
        gm.graph.erase_node(conv_node)

    # Check graph is still well-formed after transform
    gm.graph.lint()
    gm.recompile()

    return gm


class QuantizationMarker(nn.Module):
    def __init__(self, dim: tuple[int, ...] | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("min", None)
        self.register_buffer("max", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # TODO: quantiles
        current_min = torch.amin(input, dim=self.dim).detach()
        current_max = torch.amax(input, dim=self.dim).detach()

        if self.min is None:
            self.min = current_min
        else:
            self.min = torch.minimum(self.min, current_min)
        if self.max is None:
            self.max = current_max
        else:
            self.max = torch.maximum(self.max, current_max)

        return input


def insert_quantization_markers(m: fx.GraphModule) -> fx.GraphModule:
    """Insert activation markers in an independent copy of the input graph module."""
    gm = deepcopy(m)
    modules = dict(gm.named_modules())

    # Max pooling does not need a separate quantization boundary: it can operate
    # using the scale and zero point of the activation that feeds it.
    skipped_module_types = (nn.MaxPool2d,)

    for node in list(gm.graph.nodes):
        if node.op == "call_module":
            module = modules[cast(str, node.target)]
            if isinstance(module, skipped_module_types):
                continue
        elif node.op != "placeholder":
            continue

        marker_name = f"quantization_marker_{node.name}"
        gm.add_module(marker_name, QuantizationMarker())

        # Capture the existing users first so the marker does not get rewritten
        # to consume its own output.
        users = list(node.users)
        with gm.graph.inserting_after(node):
            marker_node = gm.graph.call_module(marker_name, args=(node,))
        for user in users:
            user.replace_input_with(node, marker_node)

    gm.graph.lint()
    gm.recompile()

    return gm


INT8_MIN = -128
INT8_MAX = 127
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


def quantize(values: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor):
    return torch.round(values / scale + zero).clamp(INT8_MIN, INT8_MAX).to(torch.int8)


def quantize32(values: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor):
    # Float64 represents the int32 clipping bounds exactly.
    return (
        torch.round(values.to(torch.float64) / scale + zero)
        .clamp(INT32_MIN, INT32_MAX)
        .to(torch.int32)
    )


def dequantize(values: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor):
    # Convert before subtracting: the centered value can exceed the int8 range.
    return scale * (values.to(scale.dtype) - zero)


def quantize_weights(
    weights: torch.Tensor,
    axis: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically quantize weights per tensor or per channel.

    `axis` follows the convention used by PyTorch's per-channel quantization
    APIs. For example, `axis=0` produces one scale for each output channel in
    a linear or convolution weight tensor. `axis=None` uses one scale for the
    entire tensor.
    """
    if axis is None:
        min = torch.amin(weights)
        max = torch.amax(weights)
    else:
        if not 0 <= axis < weights.dim():
            raise ValueError(f"axis must be between 0 and {weights.dim() - 1}")
        reduce_dims = tuple(
            dimension for dimension in range(weights.dim()) if dimension != axis
        )
        min = torch.amin(weights, dim=reduce_dims, keepdim=True)
        max = torch.amax(weights, dim=reduce_dims, keepdim=True)

    # Symmetric quantization
    min = torch.minimum(min, -max)
    max = torch.maximum(-min, max)

    quant_min = INT8_MIN
    quant_max = INT8_MAX

    scale = (max - min) / (quant_max - quant_min)
    # Mostly defensive, shouldn't happen in practice
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    weight_zero = torch.tensor(0, dtype=torch.int8, device=weights.device)
    int_weights = quantize(weights, scale, weight_zero)

    if axis is not None:
        scale = scale.reshape(weights.size(axis))

    return int_weights, scale


def get_activation_quantization_params(
    marker: QuantizationMarker,
) -> tuple[torch.Tensor, torch.Tensor]:
    if marker.min.numel() != 1 or marker.max.numel() != 1:
        raise ValueError("only per-tensor activation quantization is supported")

    # Stretch range to include 0 to avoid z being out of int8 range.
    # See https://confluence.cornell.edu/spaces/cev/pages/767033389/On+Quantized+CNN+Inference
    # for a brief explanation.
    zero_float = torch.zeros((), dtype=marker.min.dtype, device=marker.min.device)
    observed_min = torch.minimum(marker.min, zero_float)
    observed_max = torch.maximum(marker.max, zero_float)

    scale = (observed_max - observed_min) / (INT8_MAX - INT8_MIN)
    if scale.item() == 0:
        # I don't think this will happen at all in practice, just defensive (chat written)
        scale = torch.ones_like(scale)

    zero = torch.round(INT8_MIN - observed_min / scale).clamp(INT8_MIN, INT8_MAX)
    return scale, zero.type(torch.int8)


class QuantizedLinear(nn.Module):
    SCALE_APPROX_BITS = 31

    def __init__(
        self,
        weights: torch.Tensor,  # int8
        bias: torch.Tensor,  # int32
        w_scale: torch.Tensor,  # float
        x_scale: torch.Tensor,  # float
        x_zero: torch.Tensor,  # int8
        y_scale: torch.Tensor,  # float
        y_zero: torch.Tensor,  # int8
    ) -> None:
        super().__init__()
        self.register_buffer("weights", weights)

        correction = (
            torch.sum(weights.type(torch.int32), dim=1, dtype=torch.int32) * x_zero
        )
        self.register_buffer("full_bias", bias - correction)

        final_scale = (w_scale * x_scale) / y_scale
        approx_m = torch.round(final_scale * (2**QuantizedLinear.SCALE_APPROX_BITS))
        assert torch.all((0 <= approx_m) & (approx_m <= INT32_MAX))
        self.register_buffer("approx_m", approx_m.type(torch.int32))

        self.register_buffer("y_zero", y_zero)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        int32_accum = (
            torch.matmul(
                x.type(torch.int32), self.weights.type(torch.int32).transpose(0, 1)
            )
            + self.full_bias
        )

        m_accum = self.approx_m.type(torch.int64) * int32_accum.type(torch.int64)
        rounding_bit = (m_accum >> (QuantizedLinear.SCALE_APPROX_BITS - 1)) & 1
        scaled = (m_accum >> QuantizedLinear.SCALE_APPROX_BITS) + rounding_bit
        requantized = torch.clip(scaled + self.y_zero, INT8_MIN, INT8_MAX).type(
            torch.int8
        )

        return requantized


class QuantizedConvRelu2d(nn.Module):
    SCALE_APPROX_BITS = 31

    def __init__(
        self,
        weights: torch.Tensor,  # int8[out, in, h, w]
        bias: torch.Tensor,  # int32[out]
        w_scale: torch.Tensor,  # float[out]
        x_scale: torch.Tensor,  # float[0]
        x_zero: torch.Tensor,  # int8[0]
        y_scale: torch.Tensor,  # float[0]
        y_zero: torch.Tensor,  # int8[0]
    ) -> None:
        super().__init__()
        self.register_buffer("weights", weights)

        kernel_h = weights.size(2)
        kernel_w = weights.size(3)
        assert kernel_h % 2 == 1 and kernel_w % 2 == 1, (
            "only odd convolution kernel dimensions are supported"
        )

        correction = (
            torch.sum(weights.type(torch.int32), dim=(1, 2, 3), dtype=torch.int32)
            * x_zero
        )
        self.register_buffer("full_bias", bias - correction)

        final_scale = (w_scale * x_scale) / y_scale
        approx_m = torch.round(final_scale * (2**QuantizedConvRelu2d.SCALE_APPROX_BITS))
        assert torch.all((0 <= approx_m) & (approx_m <= INT32_MAX))
        self.register_buffer("approx_m", approx_m.type(torch.int32))

        self.register_buffer("x_zero", x_zero)
        self.register_buffer("y_zero", y_zero)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_n = x.size(0)
        out_c = self.weights.size(0)
        out_h = x.size(2)
        out_w = x.size(3)

        kernel_h = self.weights.size(2)
        kernel_w = self.weights.size(3)

        int32_accum = (
            self.full_bias.reshape(1, -1, 1, 1)
            .expand([out_n, out_c, out_h, out_w])
            .clone()
        )

        padding_h, padding_w = kernel_h // 2, kernel_w // 2
        padded = torch.nn.functional.pad(
            x, [padding_h, padding_w, padding_h, padding_w], value=self.x_zero
        )

        for c, kernel in enumerate(self.weights):
            for h in range(out_h):
                for w in range(out_w):
                    window = padded[
                        :,
                        :,
                        h : h + kernel_h,
                        w : w + kernel_w,
                    ]
                    convolved = torch.sum(
                        window.type(torch.int32) * kernel.type(torch.int32),
                        dim=(1, 2, 3),
                    )
                    int32_accum[:, c, h, w] += convolved

        relu = torch.maximum(int32_accum, torch.tensor(0, dtype=torch.int32))

        expanded_approx_m = self.approx_m.reshape(-1, 1, 1)

        m_relu = expanded_approx_m.type(torch.int64) * relu.type(torch.int64)
        rounding_bit = (m_relu >> (QuantizedConvRelu2d.SCALE_APPROX_BITS - 1)) & 1
        scaled = (m_relu >> QuantizedConvRelu2d.SCALE_APPROX_BITS) + rounding_bit
        requantized = torch.clip(scaled + self.y_zero, INT8_MIN, INT8_MAX).type(
            torch.int8
        )

        return requantized


class QuantizeActivation(nn.Module):
    """
    Used to quantize activations before they enter the network so we can
    still support a float32 interface. Not very important.
    """

    def __init__(self, scale: torch.Tensor, zero: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("zero", zero)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return quantize(input, self.scale, self.zero)


def quantize_layer_parameters(
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    x_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    int_weights, w_scale = quantize_weights(weights, axis=0)

    if bias is None:
        int_bias = torch.zeros(
            weights.size(0), dtype=torch.int32, device=weights.device
        )
    else:
        bias_zero = torch.tensor(0, dtype=torch.int32, device=bias.device)
        int_bias = quantize32(bias, x_scale * w_scale, bias_zero)

    return int_weights, int_bias, w_scale


def quantize_calibrated(gm: fx.GraphModule) -> fx.GraphModule:
    """Quantizes an already-calibrated module"""
    root = deepcopy(gm)
    modules = dict(gm.named_modules())
    new_graph = fx.Graph()
    value_remap: dict[fx.Node, fx.Node] = {}

    # Store quantization parameters for all activations so they can be referenced
    # when constructing later modules.
    qparams: dict[fx.Node, tuple[torch.Tensor, torch.Tensor]] = {}

    def marker_qparams(node: fx.Node) -> tuple[torch.Tensor, torch.Tensor]:
        marker = modules[cast(str, node.target)]
        if not isinstance(marker, QuantizationMarker):
            raise TypeError(f"Expected a quantization marker, got {type(marker)}")
        return get_activation_quantization_params(marker)

    def output_qparams(node: fx.Node) -> tuple[torch.Tensor, torch.Tensor]:
        marker_users = [
            user
            for user in node.users
            if user.op == "call_module"
            and isinstance(modules[cast(str, user.target)], QuantizationMarker)
        ]
        if len(marker_users) != 1:
            raise ValueError(
                f"Expected exactly one output marker for {node.name}, "
                f"got {len(marker_users)}"
            )
        return marker_qparams(marker_users[0])

    for node in list(gm.graph.nodes):
        if node.op == "call_module":
            module = modules[cast(str, node.target)]
            input_node = cast(fx.Node, node.args[0])

            if isinstance(module, QuantizationMarker):
                scale, zero = marker_qparams(node)
                qparams[node] = (scale, zero)
                if input_node.op == "placeholder":
                    # Quantize incoming activations so we can accept floats but run them
                    # through the quantized model.
                    root.set_submodule(
                        cast(str, node.target), QuantizeActivation(scale, zero)
                    )
                    value_remap[node] = new_graph.node_copy(
                        node, lambda argument: value_remap[argument]
                    )
                else:
                    value_remap[node] = value_remap[input_node]
                continue

            if isinstance(module, ConvRelu2d):
                x_scale, x_zero = qparams[input_node]
                y_scale, y_zero = output_qparams(node)
                int_weights, int_bias, w_scale = quantize_layer_parameters(
                    module.conv.weight.detach(), module.conv.bias, x_scale
                )
                root.set_submodule(
                    cast(str, node.target),
                    QuantizedConvRelu2d(
                        weights=int_weights,
                        bias=int_bias,
                        w_scale=w_scale,
                        x_scale=x_scale,
                        x_zero=x_zero,
                        y_scale=y_scale,
                        y_zero=y_zero,
                    ),
                )
                qparams[node] = (y_scale, y_zero)
            elif isinstance(module, nn.Linear):
                x_scale, x_zero = qparams[input_node]
                y_scale, y_zero = output_qparams(node)
                int_weights, int_bias, w_scale = quantize_layer_parameters(
                    module.weight.detach(), module.bias, x_scale
                )
                root.set_submodule(
                    cast(str, node.target),
                    QuantizedLinear(
                        weights=int_weights,
                        bias=int_bias,
                        w_scale=w_scale,
                        x_scale=x_scale,
                        x_zero=x_zero,
                        y_scale=y_scale,
                        y_zero=y_zero,
                    ),
                )
                qparams[node] = (y_scale, y_zero)
            elif isinstance(module, nn.MaxPool2d):
                qparams[node] = qparams[input_node]
            else:
                raise TypeError(
                    f"Unsupported module in quantized graph: {type(module).__name__}"
                )

            value_remap[node] = new_graph.node_copy(
                node, lambda argument: value_remap[argument]
            )
            continue

        value_remap[node] = new_graph.node_copy(
            node, lambda argument: value_remap[argument]
        )
        if node.op == "call_function":
            input_node = cast(fx.Node, node.args[0])
            qparams[node] = qparams[input_node]

    result = fx.GraphModule(root, new_graph)
    result.graph.lint()
    result.recompile()

    return result


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=base_dir / "mnist_cnn.pt")
    parser.add_argument("--output", type=Path, default=base_dir / "mnist_cnn_int8.pt")
    parser.add_argument("--data-dir", type=Path, default=base_dir / "data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--calibration-batches", type=int, default=100)
    parser.add_argument("--measure-accuracy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.calibration_batches < 1:
        raise ValueError("--calibration-batches must be at least 1")

    calibration_loader = make_loader(args.data_dir, args.batch_size, train=True)
    test_loader = make_loader(args.data_dir, args.batch_size, train=False)

    fp32_model = SimpleCNN().cpu().eval()
    fp32_model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )

    gm = torch.fx.symbolic_trace(fp32_model)
    gm.graph.print_tabular()
    print("---------------------------------")
    fused_model = fuse_conv_relu(gm)
    fused_model.graph.print_tabular()

    marked = insert_quantization_markers(fused_model)
    calibrate(marked, calibration_loader, batches=args.calibration_batches)

    print("---------------------------------")
    quantized_model = quantize_calibrated(marked)
    quantized_model.graph.print_tabular()

    if args.measure_accuracy:
        fp32_accuracy = accuracy(fp32_model, test_loader)
        fused_accuracy = accuracy(fused_model, test_loader)
        assert fp32_accuracy == fused_accuracy

        int8_accuracy = accuracy(quantized_model, test_loader)

        print(f"fp32_test_accuracy = {fp32_accuracy:.2%}")
        print(f"int8_test_accuracy = {int8_accuracy:.2%}")

    torch.save(quantized_model.state_dict(), args.output)


if __name__ == "__main__":
    main()
