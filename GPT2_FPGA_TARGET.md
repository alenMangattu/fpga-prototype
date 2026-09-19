# Exact GPT-2 inference target

The exact proof target is a GPT-2 pre-LayerNorm block:

1. `u = LayerNorm(x, gamma1, beta1, epsilon)`
2. `Q = uWq + bq`, `K = uWk + bk`, `V = uWv + bv`
3. `P = softmax((QK^T / sqrt(d_head)) + causal_mask)`
4. `r = x + (PV)Wo + bo`
5. `v = LayerNorm(r, gamma2, beta2, epsilon)`
6. `y = r + GELU_GPT2(vW1 + b1)W2 + b2`

`rtl/gpt2_block_exact_sim.sv` is the executable, bit-rounded FP32 golden
model. It includes stable causal softmax and GPT-2's tanh-form GELU. It is not
synthesizable because it deliberately uses `shortreal`, `$sqrt`, `$exp`, and
`$tanh` as the verification oracle.

`rtl/fp32_math_ip.sv` defines the ready/valid boundary for division, square
root, exponential, and tanh. In simulation it supplies behavioral FP32 math.
For synthesis, `fp32_math_vendor` must be replaced with generated FPGA IP.

The matrix engines and controller must tolerate arbitrary core latency by
waiting for `output_valid`; no fixed vendor latency may be embedded in the GPT
state machine. This allows the same computation graph to move from Efinix to
AMD or to a later ASIC implementation.

Before accepting a hardware result, compare every intermediate tensor against
the golden model: LN1, Q, K, V, scaled scores, masked scores, softmax
probabilities, attention output, first residual, LN2, GELU output, and final
residual.
