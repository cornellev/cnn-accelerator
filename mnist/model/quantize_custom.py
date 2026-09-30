import argparse
from pathlib import Path
from typing import cast

import torch
from torch import fx, nn
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


def fuse_conv_relu(
    m: torch.nn.Module, tracer_class: type = fx.Tracer
) -> fx.GraphModule:
    graph = tracer_class().trace(m)
    gm = torch.fx.GraphModule(m, graph)
    modules = dict(m.named_modules())

    for conv_node in gm.graph.nodes:
        conv_node: fx.Node

        if conv_node.op != "call_module":
            continue

        conv = modules[cast(str, conv_node.target)]
        if type(conv) is not nn.Conv2d:
            continue

        if len(conv_node.users) != 1:
            continue

        relu_node = next(iter(conv_node.users.keys()))
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


def insert_quantization_markers(
    m: torch.nn.Module, tracer_class: type = fx.Tracer
) -> fx.GraphModule:
    graph = tracer_class().trace(m)
    gm = fx.GraphModule(m, graph)
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

    gm = torch.fx.symbolic_trace(fp32_model)
    gm.graph.print_tabular()
    print("---")
    fused_model = fuse_conv_relu(fp32_model)
    fused_model.graph.print_tabular()

    fp32_accuracy = accuracy(fp32_model, test_loader)
    print(fp32_accuracy)
    fused_accuracy = accuracy(fused_model, test_loader)
    print(fused_accuracy)


if __name__ == "__main__":
    main()
