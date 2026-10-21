// ================================================================
//
// Date  : Oct 21, 2026
// Author: Utku Melemetci
//
// Parameters related to how we handle quantization.
//
// ================================================================

`ifndef _QUANT_PARAMS_SV_
`define _QUANT_PARAMS_SV_

// Mostly copied from `quantize_custom.py`
parameter WEIGHT_BITS = 8;
parameter ACTIVATION_BITS = 8;

// we have to add this before scaling, and we want scale lhs to be at
// most 27 bits (signed) so we'll let bias be up to 26 bits (since we
// add the computed 20 bit multiply-accumulate sum to it)
parameter BIAS_BITS = 26;

// we have 27x18 bit *signed* multipliers, but quant_scale_mult is unsigned
parameter SCALE_M_BITS = 17;

// we pretty much always scale down. SCALE_M_BITS - 1 allows an effective
// scale up to 1.
parameter SCALE_SHIFT = SCALE_M_BITS - 1;

`endif
