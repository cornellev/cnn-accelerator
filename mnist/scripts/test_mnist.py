from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge, with_timeout
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "model/mnist_cnn_int8.pt"
DATA_DIR = ROOT / "model/data"
IMAGE_INDICES = range(10)
CONV_NAMES = ("fused_features_0_features_1", "fused_features_3_features_4")


def drive_array(port, values: torch.Tensor) -> None:
    if values.ndim:
        for index, value in enumerate(values):
            drive_array(port[index], value)
    else:
        port.value = int(values)


@torch.inference_mode()
def run_model(model, image: torch.Tensor) -> torch.Tensor:
    return model(image)


async def run_rtl(dut, model, image: torch.Tensor) -> torch.Tensor:
    """Run the feature extractor in RTL, then apply the saved classifier."""
    for number, name in zip([1, 3], CONV_NAMES):
        conv = model.get_submodule(name)
        # start_dim=2 flattens only the kernel from 3x3 -> 9 elements, which is how we
        # do it in the RTL
        drive_array(
            getattr(dut, f"layer_{number}_weights"), conv.weights.flatten(start_dim=2)
        )
        drive_array(getattr(dut, f"layer_{number}_bias"), conv.full_bias)
        drive_array(getattr(dut, f"layer_{number}_quant_scale_mult"), conv.approx_m)
        getattr(dut, f"layer_{number}_quant_offset").value = int(conv.y_zero)

    dut.layer_1_prev_quant_offset.value = int(model.get_submodule(CONV_NAMES[0]).x_zero)
    # quantize inputs according to saved params, though technically we know this is
    # always going to be a fixed value as we force this in quantize_custom
    inputs = model.quantization_marker_inputs(image)

    received = []

    async def collect_output():
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            if dut.output_valid.value:
                received.append(
                    [dut.output_pixel[c].value.to_signed() for c in range(16)]
                )

    async def send_image():
        for value in inputs.reshape(-1).tolist():
            dut.pixel_valid.value = 1
            dut.input_pixel[0].value = value
            await FallingEdge(dut.clk)
        dut.pixel_valid.value = 0

    dut.clk.value = 0
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.pixel_valid.value = 0
    dut.input_pixel[0].value = 0

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start(start_high=False))

    await ClockCycles(dut.clk, 3)
    await FallingEdge(dut.clk)
    dut.rst_n.value = 1
    await FallingEdge(dut.clk)
    dut.start.value = 1
    await FallingEdge(dut.clk)
    dut.start.value = 0

    collector = cocotb.start_soon(collect_output())
    sender = cocotb.start_soon(send_image())

    await with_timeout(RisingEdge(dut.done), 50, "us")
    await sender
    await ClockCycles(dut.clk, 10)
    await FallingEdge(dut.clk)

    collector.cancel()
    clock.stop()

    assert len(received) == 49, f"Expected 49 spatial samples, got {len(received)}"

    # RTL emits HWC; PyTorch flattens CHW for the classifier.
    features = torch.tensor(received, dtype=torch.int8).reshape(7, 7, 16)
    features = features.permute(2, 0, 1).unsqueeze(0)
    with torch.inference_mode():
        return model.classifier(features.flatten(1))


@cocotb.test()
@cocotb.parametrize(index=IMAGE_INDICES)
async def compare_logits(dut, index):
    # Do not load unknown models! This runs arbitrary code.
    model = torch.load(CHECKPOINT, map_location="cpu", weights_only=False).eval()
    dataset = MNIST(DATA_DIR, train=False, download=False, transform=ToTensor())
    image, _ = dataset[index]
    image = image.unsqueeze(0)

    expected = run_model(model, image)
    actual = await run_rtl(dut, model, image)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def main() -> None:
    from cocotb_tools.check_results import get_results
    from cocotb_tools.runner import get_runner

    build_dir = ROOT / "target/mnist_cocotb"
    runner = get_runner("verilator")
    runner.build(
        sources=sorted((ROOT / "rtl").glob("*.sv")),
        includes=[ROOT],
        hdl_toplevel="mnist",
        build_dir=build_dir,
        build_args=["-Wno-fatal"],  # not great practice but oh well
        timescale=("1ns", "1ps"),
    )
    results = runner.test(
        hdl_toplevel="mnist",
        test_module="scripts.test_mnist",
        test_dir=build_dir,
    )
    # The standalone runner does not propagate failed assertions automatically.
    tests, failures = get_results(results)
    if failures or tests != len(IMAGE_INDICES):
        raise SystemExit(f"MNIST comparison failed: {failures}/{tests} tests failed")


if __name__ == "__main__":
    main()
