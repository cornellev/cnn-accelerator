import argparse
from pathlib import Path
import random

import torch
from PIL import Image

from model.quantize_custom import (
    ACTIVATION_BITS,
    BIAS_BITS,
    SCALE_M_BITS,
    SCALE_SHIFT,
    WEIGHT_BITS,
    QuantizedConvRelu2d,
    signed_quantization_range,
)

ROOT = Path(__file__).resolve().parents[1]
CONVOLUTION_BITS = ACTIVATION_BITS + WEIGHT_BITS + (9 - 1).bit_length()


def bits(value: int, width: int, *, signed: bool = True) -> str:
    minimum, maximum = (
        signed_quantization_range(width) if signed else (0, (1 << width) - 1)
    )

    if not minimum <= value <= maximum:
        raise ValueError(
            f"{value} does not fit in {'signed' if signed else 'unsigned'} {width} bits"
        )

    return f"{value & ((1 << width) - 1):0{width}b}"


def packed(values: list[int], width: int) -> str:
    return "".join(bits(value, width) for value in values)


def reference(
    weights: torch.Tensor,
    bias: int,
    multiplier: int,
    offset: int,
    input_zero: int = 0,
) -> QuantizedConvRelu2d:
    # A power-of-two scale makes the requested integer multiplier exact.
    return QuantizedConvRelu2d(
        weights=weights.reshape(1, 1, 3, 3).to(torch.int8),
        bias=torch.tensor([bias], dtype=torch.int32),
        w_scale=torch.tensor([multiplier / (1 << SCALE_SHIFT)], dtype=torch.float64),
        x_scale=torch.tensor(1.0, dtype=torch.float64),
        x_zero=torch.tensor(input_zero, dtype=torch.int32),
        y_scale=torch.tensor(1.0, dtype=torch.float64),
        y_zero=torch.tensor(offset, dtype=torch.int8),
    )


@torch.inference_mode()
def write_raw_vectors(path: Path, count: int, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    activation_min, activation_max = signed_quantization_range(ACTIVATION_BITS)
    weight_min, weight_max = signed_quantization_range(WEIGHT_BITS)

    with path.open("w") as file:
        for _ in range(count):
            pixels = torch.randint(
                activation_min, activation_max + 1, (9,), generator=generator
            )
            weights = torch.randint(
                weight_min, weight_max + 1, (9,), generator=generator
            )

            expected = torch.sum(
                pixels.to(torch.int32) * weights.to(torch.int32),
                dtype=torch.int32,
            )

            pixel_bits = packed(pixels.tolist(), ACTIVATION_BITS)
            weight_bits = packed(weights.tolist(), WEIGHT_BITS)
            result_bits = bits(int(expected), CONVOLUTION_BITS)

            file.write(f"{pixel_bits} {weight_bits} {result_bits}\n")


@torch.inference_mode()
def write_quantized_vectors(path: Path, count: int, seed: int) -> None:
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    activation_min, activation_max = signed_quantization_range(ACTIVATION_BITS)
    weight_min, weight_max = signed_quantization_range(WEIGHT_BITS)
    bias_min, bias_max = signed_quantization_range(BIAS_BITS)

    with path.open("w") as file:
        for _ in range(count):
            pixels = torch.randint(
                activation_min, activation_max + 1, (9,), generator=generator
            )
            weights = torch.randint(
                weight_min, weight_max + 1, (9,), generator=generator
            )

            bias = rng.choice([bias_min, bias_max, 0, rng.randint(-1024, 1024)])
            multiplier = rng.choice(
                [
                    0,
                    1,
                    1 << (SCALE_SHIFT - 1),
                    1 << SCALE_SHIFT,
                    (1 << SCALE_M_BITS) - 1,
                    rng.randrange(1 << SCALE_M_BITS),
                ]
            )
            offset = rng.randint(activation_min, activation_max)

            module = reference(weights, bias, multiplier, offset)

            # The reference pads its input; the center is the unpadded 3x3 result.
            inputs = pixels.to(torch.int8).reshape(1, 1, 3, 3)
            outputs = module(inputs)
            expected = int(outputs[0, 0, 1, 1])

            pixel_bits = packed(pixels.tolist(), ACTIVATION_BITS)
            weight_bits = packed(weights.tolist(), WEIGHT_BITS)
            params = " ".join(
                [
                    bits(int(module.full_bias[0]), BIAS_BITS),
                    bits(int(module.approx_m[0]), SCALE_M_BITS, signed=False),
                    bits(int(module.y_zero), ACTIVATION_BITS),
                ]
            )

            result_bits = bits(expected, ACTIVATION_BITS)

            file.write(f"{pixel_bits} {weight_bits} {params} {result_bits}\n")


@torch.inference_mode()
def write_picture_vectors(picture_directory: Path, vector_directory: Path) -> None:
    with Image.open(picture_directory / "calibration_board.png") as image:
        image = image.convert("L")
        width, height = image.size

        # Center the unsigned image samples into signed activation storage.
        pixels = torch.tensor(list(image.tobytes()), dtype=torch.int32)
        pixels = pixels.reshape(height, width) - 128

    weights = torch.tensor([-60, -45, -30, -15, 0, 30, 15, 45, 60], dtype=torch.int8)
    # A nonzero bias exercises bias addition in the streaming convolution path.
    module = reference(weights, bias=37, multiplier=640, offset=1, input_zero=-128)

    inputs = pixels.to(torch.int8).reshape(1, 1, height, width)
    outputs = module(inputs)
    expected = outputs[0, 0, 1:-1, 1:-1]

    picture_path = picture_directory / "calibration_board.txt"
    with picture_path.open("w") as file:
        for value in pixels.flatten().tolist():
            file.write(bits(value, ACTIVATION_BITS) + "\n")

    expected_path = vector_directory / "quantized_convolution_calibration_board_vectors.txt"
    with expected_path.open("w") as file:
        for value in expected.flatten().tolist():
            file.write(bits(value, ACTIVATION_BITS) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-vectors", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.num_vectors <= 0:
        parser.error("--num-vectors must be positive")

    torch.set_num_threads(1)

    directory = ROOT / "tb" / "vectors"
    directory.mkdir(parents=True, exist_ok=True)

    write_raw_vectors(directory / "convolve_vectors.txt", args.num_vectors, args.seed)
    write_quantized_vectors(
        directory / "quantized_convolve_vectors.txt", args.num_vectors, args.seed
    )
    write_picture_vectors(ROOT / "tb" / "picture", directory)


if __name__ == "__main__":
    main()
