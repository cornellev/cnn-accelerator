# Winograd 3×3 Convolution: How We Implemented It

**Module:** `mnist/rtl/winograd_3x3_conv.sv`
**Reference model / vectors:** `mnist/scripts/winograd_3x3_vectors.py`
**Testbench:** `mnist/tb/winograd_3x3_conv_tb.sv` (100,000 random vectors plus corner cases, 0 errors)
**Status:** Standalone and verified. Not yet integrated into `convolve_3x3_fsm`.

---

## 1. Summary

`winograd_3x3_conv` computes a 2×2 block of 3×3 convolution outputs from a 4×4 pixel tile using **16 multiplies instead of 36**. That is 2.25× fewer multipliers per output. It is a combinational drop-in for four `convolve_3x3` instances. The algorithm is Winograd F(2×2, 3×3).

The catch is that the multiplies are wider: 12-bit × 11-bit instead of 8-bit × 8-bit. On the Kria KV260 that matters, because INT8 packing on the DSP48E2 already lets direct convolution do two 8-bit MACs per DSP. Section 6 compares the two approaches.

---

## 2. Why Winograd uses fewer multiplies

Direct convolution of a 4×4 tile with a 3×3 kernel computes four overlapping 3×3 windows. That is 4 × 9 = 36 multiplies, even though there are only 16 distinct pixels. The centre four pixels are multiplied four times each.

Winograd removes that redundancy. In one dimension, F(2,3) produces 2 outputs from 4 pixels (d0..d3) and 3 weights (g0..g2):

```
Direct (6 multiplies):
  y0 = d0*g0 + d1*g1 + d2*g2
  y1 = d1*g0 + d2*g1 + d3*g2

Winograd (4 multiplies):
  m1 = (d0 - d2) * g0
  m2 = (d1 + d2) * (g0 + g1 + g2) / 2
  m3 = (d2 - d1) * (g0 - g1 + g2) / 2
  m4 = (d1 - d3) * g2

  y0 = m1 + m2 + m3
  y1 = m2 - m3 - m4
```

Each product multiplies a *sum of pixels* by a *sum of weights*. Expanding them gives some unwanted cross terms, but those terms appear with opposite signs in different products, so they cancel when the products are added.

Three things make this cheap:

- **Weight sums are free at runtime.** Weights are fixed after training, so the weight-side sums are computed once offline and stored.
- **Pixel sums are only adders.** All transform coefficients are 0, +1 or −1.
- **Only the 4 products need multipliers.**

**Where the formulas come from.** Convolution is polynomial multiplication. Treat the kernel as g(x) = g0 + g1·x + g2·x². The weight sums above are g evaluated at x = 0, 1, −1 and ∞ (g(∞) means the leading coefficient, g2). Evaluating at 4 points, multiplying pointwise and interpolating back gives the result with 4 multiplies. FFT convolution is the same recipe evaluated at the complex roots of unity (1, −i, −1, i). The complex points make each product a complex multiply (3–4 real multiplies), which is why FFT loses to Winograd for small kernels.

---

## 3. The 2D algorithm as implemented

Applying the 1D transform to rows and then to columns gives F(2×2, 3×3):

```
Y = A^T [ (G' g G'^T) ⊙ (B^T d B) ] A      then   y = Y >>> 2
```

where `d` is the 4×4 tile, `g` is the 3×3 kernel and `⊙` is elementwise multiply.

```
B^T = [ 1  0 -1  0 ]     G' = 2G = [ 2  0  0 ]     A^T = [ 1  1  1  0 ]
      [ 0  1  1  0 ]               [ 1  1  1 ]           [ 0  1 -1 -1 ]
      [ 0 -1  1  0 ]               [ 1 -1  1 ]
      [ 0  1  0 -1 ]               [ 0  0  2 ]
```

| Stage | RTL signals | Operation | Cost |
|---|---|---|---|
| Input transform, rows | `bt[]` | B^T · d | 16 adds |
| Input transform, columns | `v[]` | (B^T d) · B | 16 adds |
| Kernel transform | `u_weights[]` input | U' = G' g G'^T | Offline, once per kernel |
| Elementwise multiply | `m[]` | M = U' ⊙ V | **16 multiplies** |
| Output transform, rows | `at[]` | A^T · M | 16 adds |
| Output transform, columns | `y[]` | (A^T M) · A | 8 adds |
| Rescale | `convolution[]` | Y >>> 2 | Wiring only |

**Why G' = 2G and the final shift.** The textbook G matrix contains ½ entries. We scale it by 2 so that U' = G' g G'^T = 4·(G g G^T) is always an integer. The output comes out exactly 4× too large, and `>>> 2` removes that with no rounding error. The result is **bit-exact** with direct convolution.

### Port and internal widths (current RTL)

| Signal | Width | Notes |
|---|---|---|
| `tile` | 16 × 8-bit **unsigned** | Zero-extended into `d` (11-bit signed) |
| `u_weights` | 16 × 12-bit signed | 9 int8 weights × 4 needs up to ±1152 |
| `v` | 11-bit signed | Sums of up to 4 pixels with ±1 coefficients |
| `m` | 23-bit signed | 12 × 11 product |
| `y` | 25-bit signed | Sums of 9 products |
| `convolution` | 4 × 20-bit signed | Same width as `convolve_3x3` |

### Verification

`winograd_3x3_vectors.py` computes the expected output with both a direct reference and a Python model of the Winograd datapath. It checks that the two agree, then writes 100,000 random vectors plus corner cases into `tb/vectors/winograd_vectors.txt`. The corner cases are all-max pixels with max-positive, max-negative and alternating-sign kernels. `winograd_3x3_conv_tb.sv` runs all of them with 0 errors.

---

## 4. Limitations and open issues

**Out of date with the PR #4 width changes.** `main` now uses **signed int8 activations** (`quant_params.sv`). `winograd_3x3_conv` still zero-extends unsigned pixels, and its vector script still generates unsigned tiles. Before this module can replace `convolve_3x3` in the quantized path, it needs three changes:
- Sign-extend `tile` instead of zero-extending it.
- Shrink `d`/`bt`/`v`. With signed pixels, `v` fits in 10 bits.
- Regenerate the vectors with signed pixels.

**Wider multipliers.** The products are 12 × 11 bits (12 × 10 after the signed fix), compared with 8 × 8 for direct. Each one still fits in a single DSP48E2 (27 × 18), but they are too wide for INT8 packing (Section 6). This is the main limitation on the KV260.

**More weight storage.** Each kernel is stored as 16 × 12-bit = 192 bits, compared with 9 × 8-bit = 72 bits for direct. That is 2.7× more weight memory (BRAM/URAM or registers). Note that the `.mem` export from `quantize_custom.py` stores raw int8 weights, so U' would have to be computed either at export time or by a small kernel-transform unit on chip.

**Only 3×3, stride 1.** The transform matrices are specific to a 3×3 kernel with stride 1. Our MNIST network only uses 3×3 stride-1 convolutions, so this is fine here, but it does not generalise.

**Tiled input and output.** The module needs a 4×4 window and produces a 2×2 block, stepping by 2 pixels. Integrating it into the FSM would require:
- 3 row buffers instead of 2, and a 4×4 window register instead of 3×3.
- Output blocks every other row and column, so the output stream is no longer one value per cycle.
- Handling for odd output sizes, by padding or by a direct-convolution fallback on the edge.

The good news is that every conv in our model is followed by `MaxPool2d(2)`. With `padding=1`, the 2×2 output blocks line up exactly with the pooling windows, so the pooling can be fused directly after the output transform.

**Critical path.** The module is fully combinational: 2 adder levels, then a multiply, then 2 more adder levels. At useful KV260 clock rates (200–300 MHz) it needs pipeline registers. The natural cut points are after `v`, after `m` (inside the DSP's M register) and after `y`.

**Bigger tiles don't scale cleanly.** F(4×4, 3×3) would need 36 multiplies for 16 outputs, which is 4× fewer than direct. But it evaluates at extra points such as ±2 and ±½. The transform constants grow, intermediate widths grow by several bits, and the transforms need real multiplies by constants. F(2×2, 3×3) is the sweet spot for 8-bit integer hardware.

**Zero points and bias.** Winograd is linear, so the existing `full_bias` approach still works: the input-zero-point correction is already folded into the bias. The bias is added after the output transform and then goes through the unchanged `quantized_scale`.

---

## 5. Kria KV260 resources

The KV260 uses the Zynq UltraScale+ **XCK26**:

| Resource | XCK26 |
|---|---|
| DSP48E2 slices | 1,248 |
| LUTs | ~117k |
| Block RAM / UltraRAM | 144 BRAM (36 Kb) + 64 URAM (288 Kb), about 26.6 Mb total on-chip |

Each DSP48E2 has a **27 × 18 signed multiplier**, a **27-bit pre-adder** in front of it and a 48-bit accumulator. Our `quant_params.sv` already targets this size: the scale multiply is 27 × 18.

---

## 6. Winograd vs. 8-bit-packed direct convolution on the KV260

### How INT8 packing works

AMD/Xilinx WP486 ("Deep Learning with INT8 Optimization on Xilinx Devices") describes how to get **two int8 × int8 MACs from one DSP48E2**, as long as the two multiplies share one operand:

```
(w1 << 18 + w2) * x  =  (w1*x) << 18  +  (w2*x)
```

The pre-adder builds `w1·2^18 + w2` on the 27-bit port, and `x` goes on the 18-bit port.
- The low 18 bits hold `w2·x` and the upper bits hold `w1·x`.
- A small correction is needed because a negative low product borrows 1 from the high field. Add the low field's sign bit back.
- According to WP486, the 18-bit low field has room to accumulate about 7 products before it overflows into the high field (7 × 2^14 < 2^17).

**This fits our design.** In `convolve_3x3_fsm`, all `KERNEL_COUNT` kernels see the same pixel window. So for each pixel we can pack **weights from two different kernels** and share the pixel as `x`. One DSP then serves two kernels. That doubles the throughput of direct convolution per DSP, with no change to the dataflow or the weight format.

### Why Winograd can't also pack

The Winograd products are 12 × 10–11 bits, about 22 bits wide. Two such products, plus guard bits, don't fit in a 27 × 18 multiplier, so Winograd gets **one multiply per DSP**. As a result, most of Winograd's advantage disappears against packed direct convolution.

### DSP cost per output pixel

These figures count multipliers only. The requantize multiply (1 DSP per output per kernel) is the same for every approach and is not included.

| | Direct (1 MAC/DSP) | Direct + INT8 packing (2 MACs/DSP) | Winograd F(2×2,3×3) |
|---|---|---|---|
| Per 2×2 block, one kernel | 36 | **18** | **16** |
| Layer 1 (1 → 8 ch), per output pixel | 72 | 36 | 32 |
| Layer 2 (8 → 16 ch), per output pixel | 1,152 | 576 | 512 |
| Multiplier width | 8 × 8 | 2 × (8 × 8) in 27 × 18 | 12 × 10/11 |
| Weight storage per 3×3 kernel | 72 bits | 72 bits | 192 bits |
| Extra adders | Accumulation only | Accumulation + sign correction | 32 input-transform adds per tile per input channel, 24 output-transform adds per tile per output channel |
| Window / buffers | 3×3, 2 row buffers | 3×3, 2 row buffers | 4×4, 3 row buffers |
| Output pattern | 1 per cycle | 1 per cycle | 2×2 block every 2 columns |
| Fits current FSM | Yes | Yes, small change | Needs FSM rework |
| Bit-exact | Yes | Yes, with correction | Yes |

For layer 2, Winograd adds up the products across the 8 input channels while still in the transform domain. That means the output transform runs once per output channel, not once per input-output pair, so its add cost stays small.

### How to read the comparison

- Against **unpacked direct**, Winograd saves 2.25× DSPs.
- Against **packed direct**, Winograd saves only about **11%** (16 vs 18 per block). In exchange it needs 2.7× the weight storage, a 4×4 window with an extra row buffer, a reworked FSM and output stream, and on-chip or export-time kernel transforms.
- **We are not DSP-bound right now.** The FSM takes one pixel per clock. Layer 1 needs at most 72 DSPs unpacked, which is about 6% of the 1,248 available. Layer 2 fully parallel would need 1,152 unpacked (92%), 576 packed or 512 Winograd. Packing alone brings layer 2 comfortably within budget.
- **Multiplies can also go in LUTs.** An 8 × 8 multiplier is roughly 60–70 LUTs, so with ~117k LUTs some multiplies can be moved out of DSPs if needed. Wider Winograd products are much more expensive in LUTs.

### Recommendation

1. **Make INT8-packed direct convolution the main datapath.** It reuses the existing FSM, row buffers, weight export and testbenches. It needs about the same DSPs as Winograd and has a much smaller integration risk.
2. **Keep Winograd as a verified alternative** for cases where DSPs become the bottleneck. Examples are a larger network or many layers instantiated in parallel, where its 11% saving starts to matter, or a design where fusing the 2×2 output block with MaxPool is attractive.
3. **If we pursue Winograd**, the next steps are: convert it to signed int8 pixels, pipeline it, compute U' in the export script and add a tiled FSM with fused max-pooling.

---

## 7. References

- Lavin & Gray, "Fast Algorithms for Convolutional Neural Networks," CVPR 2016 (Winograd F(m×m, r×r) for CNNs).
- AMD/Xilinx WP486, "Deep Learning with INT8 Optimization on Xilinx Devices" (INT8 packing on DSP48E2).
- AMD UG579, "UltraScale Architecture DSP Slice User Guide" (DSP48E2 ports and pre-adder).
- Interactive explainer: `mnist/docs/winograd_explainer.html`.
