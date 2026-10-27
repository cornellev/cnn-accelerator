"""Export an MNIST sample or grayscale image and its first-layer feature maps.

Entirely written by chat but it's just for testing so I kinda don't care
Have not reviewed for code quality but seems to be correct
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets, transforms

from quantize_custom import QuantizeActivation, QuantizedConvRelu2d


def load_first_layer(checkpoint: Path) -> torch.nn.Sequential:
    """Restore the input quantizer and first Conv/ReLU from a custom checkpoint."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    quantizer_name = "quantization_marker_inputs"
    conv_name = "fused_features_0_features_1"
    quantizer = QuantizeActivation(
        state[f"{quantizer_name}.scale"], state[f"{quantizer_name}.zero"]
    )
    conv_state = {
        key.removeprefix(f"{conv_name}."): value
        for key, value in state.items()
        if key.startswith(f"{conv_name}.")
    }
    weights = conv_state["weights"]
    if weights.shape != (8, 1, 3, 3) or weights.dtype != torch.int8:
        raise ValueError("Expected an int8 first convolution with shape (8, 1, 3, 3)")

    # Original floating-point weight/output scales are not saved. Initialize
    # placeholders, then restore the exact integer inference buffers below.
    conv = QuantizedConvRelu2d(
        weights=weights,
        bias=torch.zeros(8, dtype=torch.int32),
        w_scale=torch.zeros(8),
        x_scale=quantizer.scale,
        x_zero=conv_state["x_zero"],
        y_scale=torch.tensor(1.0),
        y_zero=conv_state["y_zero"],
    )
    conv.load_state_dict(conv_state, strict=True)
    return torch.nn.Sequential(quantizer, conv).eval()


def write_binary_images(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    directory: Path,
    *,
    input_name: str = "input.txt",
    output_name: str = "output.txt",
) -> None:
    """Write row-major pixels. Each output line is one 64-bit word with
    channel 0 in the top byte and channel 7 in the bottom byte."""
    if inputs.dtype != torch.int8 or outputs.dtype != torch.int8:
        raise ValueError("Input and output images must be int8 tensors")
    if inputs.ndim != 2 or outputs.shape != (8, *inputs.shape):
        raise ValueError("Expected input [height, width] and output [8, height, width]")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / input_name).write_text(
        "".join(f"{value & 0xFF:08b}\n" for value in inputs.reshape(-1).tolist())
    )
    pixels = outputs.permute(1, 2, 0).reshape(-1, 8).tolist()
    (directory / output_name).write_text(
        "".join(
            "".join(f"{value & 0xFF:08b}" for value in pixel) + "\n" for pixel in pixels
        )
    )


def load_image(path: Path) -> torch.Tensor:
    """Preserve image dimensions and normalize grayscale camera bytes to [0, 1]."""
    with Image.open(path) as image:
        image = image.convert("L")
        width, height = image.size
        pixels = torch.tensor(list(image.tobytes()), dtype=torch.float32)
        return pixels.reshape(1, height, width) / 255


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=base_dir / "mnist_cnn_int8.pt"
    )
    parser.add_argument("--data-dir", type=Path, default=base_dir / "data")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--index", type=int, help="MNIST test image index (default: 0)")
    source.add_argument(
        "--image", type=Path, help="Image to convert to grayscale; no resizing"
    )
    parser.add_argument("--output-dir", type=Path, default=base_dir / "image_export")
    parser.add_argument("--input-name", default="input.txt")
    parser.add_argument("--output-name", default="output.txt")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    model = load_first_layer(args.checkpoint)
    if args.image is not None:
        image = load_image(args.image)
        description = str(args.image)
    else:
        dataset = datasets.MNIST(
            args.data_dir, train=False, download=True, transform=transforms.ToTensor()
        )
        index = args.index if args.index is not None else 0
        if not 0 <= index < len(dataset):
            raise ValueError(f"--index must be between 0 and {len(dataset) - 1}")
        image, label = dataset[index]
        description = f"MNIST test image {index} (label {label})"
    inputs = model[0](image.unsqueeze(0))
    outputs = model[1](inputs)
    write_binary_images(
        inputs[0, 0],
        outputs[0],
        args.output_dir,
        input_name=args.input_name,
        output_name=args.output_name,
    )
    print(
        f"Exported {description} ({image.shape[-1]}x{image.shape[-2]}) to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
