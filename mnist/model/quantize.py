"""Apply PT2E post-training static INT8 quantization to the MNIST CNN."""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.fx import GraphModule, Node
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN

# PT2E quantization is pure Python for this model; skip unrelated optional TorchAO
# CUDA extensions, which may not be present in a CPU-only installation.
os.environ.setdefault("TORCHAO_FORCE_SKIP_LOADING_SO_FILES", "1")
logging.getLogger("torchao").setLevel(logging.ERROR)
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"torchao\..*")

import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
)


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


def dynamic_batch_shape() -> tuple[dict[int, torch.export.Dim]]:
    return ({0: torch.export.Dim("batch", min=1)},)


def export_for_pt2e(model: nn.Module, example: torch.Tensor) -> GraphModule:
    """Capture an ATen graph with a dynamic batch dimension."""
    return torch.export.export(
        model,
        (example,),
        dynamic_shapes=dynamic_batch_shape(),
    ).module()


def get_attr(model: GraphModule, node: Node) -> Any:
    if node.op != "get_attr":
        raise TypeError(f"Expected get_attr node, got {node.op}: {node}")
    value: Any = model
    for component in str(node.target).split("."):
        value = getattr(value, component)
    return value


def layer_name(node: Node) -> str:
    """Recover the original module path retained in export metadata."""
    module_stack = node.meta.get("nn_module_stack", {})
    paths = [path for path, _module_type in module_stack.values() if path]
    return paths[-1] if paths else node.name


def print_quantized_parameters(model: GraphModule) -> None:
    """Print integer weights/biases and every weight/activation qparam."""
    torch.set_printoptions(profile="full", linewidth=160)
    layer_ops = {torch.ops.aten.conv2d.default, torch.ops.aten.linear.default}

    print("\n=== INTEGER LAYER PARAMETERS AND WEIGHT QPARAMS ===")
    for node in model.graph.nodes:
        if node.op != "call_function" or node.target not in layer_ops:
            continue

        input_dequant = node.args[0]
        weight_dequant = node.args[1]
        bias_node = node.args[2]
        if not isinstance(input_dequant, Node) or not isinstance(weight_dequant, Node):
            continue

        weight = get_attr(model, weight_dequant.args[0])
        scales = get_attr(model, weight_dequant.args[1])
        zero_points = get_attr(model, weight_dequant.args[2])
        axis, quant_min, quant_max, dtype = weight_dequant.args[3:7]
        input_scale = float(input_dequant.args[1])
        bias = get_attr(model, bias_node) if isinstance(bias_node, Node) else None

        name = layer_name(node)
        print(f"\n[{name}] {node.target}")
        print(f"weight.shape = {tuple(weight.shape)}")
        print(f"weight.integer ({weight.dtype}) =\n{weight}")
        print(
            "weight.qparams = "
            f"{{'dtype': '{dtype}', 'mapping': 'symmetric', "
            f"'granularity': 'per_channel', 'axis': {axis}, "
            f"'quant_min': {quant_min}, 'quant_max': {quant_max}, "
            f"'scales': {scales.tolist()}, "
            f"'zero_points': {zero_points.tolist()}}}"
        )

        if isinstance(bias, torch.Tensor):
            integer_bias = torch.round(
                bias.detach().to(torch.float64)
                / (input_scale * scales.to(torch.float64))
            ).to(torch.int32)
            print(f"bias.fp32 (stored by PT2E) =\n{bias.detach()}")
            print(
                "bias.effective_int32 = round(bias.fp32 / "
                f"(input_scale={input_scale} * weight_scale)) =\n{integer_bias}"
            )

    print("\n=== ACTIVATION QPARAMS ===")
    quantize_target = torch.ops.quantized_decomposed.quantize_per_tensor.default
    for node in model.graph.nodes:
        if node.op != "call_function" or node.target != quantize_target:
            continue
        source, scale, zero_point, quant_min, quant_max, dtype = node.args
        source_name = source.name if isinstance(source, Node) else str(source)
        print(
            f"{source_name}: {{'dtype': '{dtype}', 'mapping': 'asymmetric', "
            f"'granularity': 'per_tensor', 'scale': {float(scale)}, "
            f"'zero_point': {int(zero_point)}, 'quant_min': {quant_min}, "
            f"'quant_max': {quant_max}}}"
        )


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=base_dir / "mnist_cnn.pt")
    parser.add_argument("--output", type=Path, default=base_dir / "mnist_cnn_int8.pt2")
    parser.add_argument("--data-dir", type=Path, default=base_dir / "data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--calibration-batches", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.calibration_batches < 1:
        raise ValueError("--calibration-batches must be at least 1")

    calibration_loader = make_loader(args.data_dir, args.batch_size, train=True)
    test_loader = make_loader(args.data_dir, args.batch_size, train=False)
    example_images, _ = next(iter(calibration_loader))

    fp32_model = SimpleCNN().cpu().eval()
    fp32_model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    fp32_accuracy = accuracy(fp32_model, test_loader)

    exported_model = export_for_pt2e(fp32_model, example_images)
    quantizer = X86InductorQuantizer().set_global(
        xiq.get_default_x86_inductor_quantization_config()
    )
    prepared_model = prepare_pt2e(exported_model, quantizer)
    calibrate(prepared_model, calibration_loader, args.calibration_batches)
    quantized_model = convert_pt2e(prepared_model)

    int8_accuracy = accuracy(quantized_model, test_loader)
    print_quantized_parameters(quantized_model)

    exported_quantized_model = torch.export.export(
        quantized_model,
        (example_images,),
        dynamic_shapes=dynamic_batch_shape(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported_quantized_model, args.output)

    print("\n=== SUMMARY ===")
    print("quantizer = X86InductorQuantizer")
    print(f"calibration_batches = {args.calibration_batches}")
    print(f"fp32_test_accuracy = {fp32_accuracy:.2%}")
    print(f"int8_test_accuracy = {int8_accuracy:.2%}")
    print(f"saved_quantized_model = {args.output}")


if __name__ == "__main__":
    main()
