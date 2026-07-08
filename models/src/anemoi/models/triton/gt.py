# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

# If Triton is missing, this backend cannot run. This also covers CPU-only
# PyTorch installs, where Triton is not available.
try:
    import triton
    import triton.language as tl
except ImportError:
    raise ValueError(
        "Error. The 'triton' backend was selected for the GraphTransformer but Triton is not installed. To use this backend please install Triton. Otherwise, select a different backend for the GraphTransformer in the models config."
    )


@triton.jit
def build_masks_and_offsets(H: tl.constexpr, C: tl.constexpr, H_pad: tl.constexpr, C_pad: tl.constexpr):
    """Return masks and flat offsets for a padded [H, C] tile.

    Triton kernels like power-of-two tiles, so H and C are padded up when
    needed. This keeps support for non-power-of-two numbers of heads and
    channels without changing the real tensor layout in memory.

    Returns a head mask, a flattened [H, C] mask, and flattened offsets into
    one compact [H, C] block per node or edge.

    If H and C are already powers of two, the mask stays trivial. If only C
    needs padding, we can skip the extra head mask.
    """

    H_mask = True
    H_C_mask = True

    if H == H_pad and C == C_pad:
        H_C_off = tl.arange(0, H * C)

    elif H == H_pad:
        C_pad_off = tl.arange(0, C_pad)[None, :]
        H_off = tl.arange(0, H)[:, None]
        # Build the [H, C_pad] mask in 2D, then flatten it because the kernel
        # loads and stores one flat [H * C] block in memory.
        H_C_mask_2d = (C_pad_off < C) & (H_off < H)
        H_C_mask = tl.reshape(H_C_mask_2d, (H * C_pad,))
        H_C_off = tl.reshape(H_off * C + C_pad_off, (H * C_pad,))

    else:
        H_pad_off = tl.arange(0, H_pad)[:, None]
        C_pad_off = tl.arange(0, C_pad)[None, :]

        H_mask = tl.arange(0, H_pad) < H
        # Same idea here: build the 2D padded mask first, then flatten it for
        # the actual 1D memory access pattern.
        H_C_mask_2d = (C_pad_off < C) & (H_pad_off < H)
        H_C_mask = tl.reshape(H_C_mask_2d, (H_pad * C_pad,))
        # Offsets still point into the real unpadded [H, C] tensor in memory.
        # The stride uses the real C because the backing tensors themselves are
        # not padded. Extra padded positions are masked out on load/store.
        H_C_off = tl.reshape(H_pad_off * C + C_pad_off, (H_pad * C_pad,))

    return H_mask, H_C_mask, H_C_off


@triton.jit
def _gt_fwd(
    Q_ptr,  # [N_dst, H, C]
    K_ptr,  # [N_src, H, C]
    V_ptr,  # [N_src, H, C]
    E_ptr,  # [M, H, C]
    MMAX_ptr,  # [N_dst, H]
    INV_L_ptr,  # [N_dst, H]
    ROW_ptr,  # [M]
    COLPTR_ptr,  # [N_dst+1]
    OUT_ptr,  # [N_dst, H, C]
    OUT_FP32_ptr,  # [N_dst, H, C]
    N_dst,
    H: tl.constexpr,
    C: tl.constexpr,
    out_dtype: tl.constexpr,
):
    pid = tl.program_id(0)
    dst_idx = pid
    if dst_idx >= N_dst:
        return

    H_pad: tl.constexpr = triton.next_power_of_2(H)
    C_pad: tl.constexpr = triton.next_power_of_2(C)
    H_mask, H_C_mask, H_C_off = build_masks_and_offsets(H, C, H_pad, C_pad)

    dst_start = dst_idx * H * C
    dst_off = dst_start + H_C_off

    neigh_start = tl.load(COLPTR_ptr + dst_idx)
    neigh_end = tl.load(COLPTR_ptr + dst_idx + 1)
    num_edges = neigh_end - neigh_start

    h_off = tl.arange(0, H_pad)
    row_off = dst_idx * H + h_off

    if num_edges == 0:
        # No incoming edges: output is zero and saved stats are zero too.
        zeros_h = tl.zeros((H_pad,), dtype=tl.float32)
        zeros_hc = tl.zeros((H_pad * C_pad,), dtype=tl.float32)
        tl.store(MMAX_ptr + row_off, zeros_h, mask=H_mask)
        tl.store(INV_L_ptr + row_off, zeros_h, mask=H_mask)
        tl.store(OUT_FP32_ptr + dst_off, zeros_hc, mask=H_C_mask)
        tl.store(OUT_ptr + dst_off, zeros_hc.to(out_dtype), mask=H_C_mask)
        return

    # Extra padded positions read as zero.
    q = tl.load(Q_ptr + dst_off, mask=H_C_mask, other=0.0).to(tl.float32).reshape((H_pad, C_pad))
    acc = tl.zeros((H_pad, C_pad), dtype=tl.float32)  # running output sum before the final divide
    l_i = tl.zeros((H_pad,), dtype=tl.float32)  # sum of attention weights
    m_i = tl.full((H_pad,), value=-float("inf"), dtype=tl.float32)  # largest score seen so far

    # Helper pointers so we do not rebuild the same edge offsets every time.
    # Walk over all incoming edges for this destination in CSC order and keep
    # the running softmax state for that destination row.
    # `COLPTR_ptr` gives the start and end of that row in the CSC edge list.
    edge_ptr = E_ptr + neigh_start * H * C + H_C_off
    e_idx = neigh_start
    qk_scale: tl.constexpr = 1.0 / tl.sqrt(float(C))

    for _ in range(num_edges):
        e = tl.load(edge_ptr, mask=H_C_mask, other=0.0).to(tl.float32).reshape((H_pad, C_pad))

        src_idx = tl.load(ROW_ptr + e_idx)
        src_off = src_idx * H * C + H_C_off
        k = tl.load(K_ptr + src_off, mask=H_C_mask, other=0.0).to(tl.float32).reshape((H_pad, C_pad))
        v = tl.load(V_ptr + src_off, mask=H_C_mask, other=0.0).to(tl.float32).reshape((H_pad, C_pad))

        k_e = k + e
        v_e = v + e

        qk = tl.sum(q * k_e, axis=-1) * qk_scale

        m_ij = tl.maximum(m_i, qk)
        alpha_ij = tl.exp(qk - m_ij)
        correction = tl.exp(m_i - m_ij)

        acc = acc * correction[:, None]
        l_i = l_i * correction

        acc = acc + alpha_ij[:, None] * v_e
        l_i = l_i + alpha_ij
        m_i = m_ij

        edge_ptr += H * C
        e_idx += 1

    # Final normalize by the softmax row sum.
    inv_l_i = 1.0 / l_i
    out_fp32 = acc * inv_l_i[:, None]

    # Save the returned output, plus an fp32 copy and the final softmax row
    # stats. Backward uses these saved values to rebuild the same probabilities
    # without rerunning the online softmax update.
    tl.store(OUT_FP32_ptr + dst_off, out_fp32.reshape(H_pad * C_pad), mask=H_C_mask)
    tl.store(OUT_ptr + dst_off, out_fp32.to(out_dtype).reshape(H_pad * C_pad), mask=H_C_mask)
    tl.store(MMAX_ptr + row_off, m_i, mask=H_mask)
    tl.store(INV_L_ptr + row_off, inv_l_i, mask=H_mask)


@triton.jit
def _gt_bwd_dst_pass(
    Q_ptr,
    K_ptr,
    V_ptr,
    E_ptr,
    MMAX_ptr,  # [N_dst, H]
    INV_L_ptr,  # [N_dst, H]
    ROW_ptr,  # [M]
    COLPTR_ptr,  # [N_dst + 1]
    OUT_FP32_ptr,  # [N_dst, H, C]
    D_OUT_ptr,  # [N_dst, H, C]
    D_Q_ptr,  # [N_dst, H, C]
    N_dst,
    H: tl.constexpr,
    C: tl.constexpr,
    out_dtype: tl.constexpr,
):
    dst_idx = tl.program_id(0)
    if dst_idx >= N_dst:
        return

    H_pad: tl.constexpr = triton.next_power_of_2(H)
    C_pad: tl.constexpr = triton.next_power_of_2(C)
    H_mask, H_C_mask, H_C_off = build_masks_and_offsets(H, C, H_pad, C_pad)

    dst_off = dst_idx * H * C + H_C_off

    neigh_start = tl.load(COLPTR_ptr + dst_idx)
    neigh_end = tl.load(COLPTR_ptr + dst_idx + 1)
    num_edges = neigh_end - neigh_start

    if num_edges == 0:
        # store D_j = <d_out, out> = 0 and dQ = 0
        zeros = tl.zeros((H_pad,), dtype=tl.float32)
        tl.store(D_ptr + dst_idx * H + tl.arange(0, H_pad), zeros, mask=H_mask)
        zeros = tl.zeros((H_pad * C_pad,), dtype=out_dtype)
        tl.store(D_Q_ptr + dst_off, zeros, mask=H_C_mask)
        return

    d_out = tl.load(D_OUT_ptr + dst_off, mask=H_C_mask, other=0.0).to(tl.float32).reshape((H_pad, C_pad))
    q = tl.load(Q_ptr + dst_off, mask=H_C_mask, other=0.0).to(tl.float32).reshape((H_pad, C_pad))
    out_j = tl.load(OUT_FP32_ptr + dst_off, mask=H_C_mask, other=0.0).to(tl.float32).reshape((H_pad, C_pad))

    h_off = tl.arange(0, H_pad)
    row_off = dst_idx * H + h_off
    m_j = tl.load(MMAX_ptr + row_off, mask=H_mask, other=float("inf")).to(tl.float32)
    inv_l_j = tl.load(INV_L_ptr + row_off, mask=H_mask, other=0.0).to(tl.float32)

    # This pass walks one destination row in CSC order, the same order used in
    # forward for that node, and accumulates the pieces needed for dQ.
    # Dj = <d_out, out> is one row-wise term shared by every incoming edge.
    Dj = tl.sum(d_out * out_j, axis=-1)
    sum_p_ke = tl.zeros((H_pad, C_pad), dtype=tl.float32)
    sum_p_dalpha_ke = tl.zeros((H_pad, C_pad), dtype=tl.float32)

    edge_ptr = E_ptr + neigh_start * H * C + H_C_off
    e_idx = neigh_start
    qk_scale: tl.constexpr = 1.0 / tl.sqrt(float(C))

    for _ in range(num_edges):
        e = tl.load(edge_ptr, mask=H_C_mask, other=0.0).to(tl.float32).reshape((H_pad, C_pad))

        src = tl.load(ROW_ptr + e_idx)
        src_off = src * H * C + H_C_off
        k = tl.load(K_ptr + src_off, mask=H_C_mask, other=0.0).to(tl.float32).reshape((H_pad, C_pad))
        v = tl.load(V_ptr + src_off, mask=H_C_mask, other=0.0).to(tl.float32).reshape((H_pad, C_pad))

        ke = k + e
        ve = v + e

        # Rebuild the forward softmax probabilities from the saved row max and
        # inverse row sum.
        s_ij = tl.sum(q * ke, axis=-1) * qk_scale
        p_ij = tl.exp(s_ij - m_j) * inv_l_j
        dalpha = tl.sum(d_out * ve, axis=-1)

        p_dalpha = p_ij * dalpha
        sum_p_ke += p_ij[:, None] * ke
        sum_p_dalpha_ke += p_dalpha[:, None] * ke

        edge_ptr += H * C
        e_idx += 1

    # store D_j and dQ
    tl.store(D_ptr + dst_idx * H + tl.arange(0, H_pad), Dj, mask=H_mask)
    tl.store(
        D_Q_ptr + dst_off,
        dq.to(out_dtype).reshape(
            H_pad * C_pad,
        ),
        mask=H_C_mask,
    )


@triton.jit
def _gt_bwd_src_pass(
    Q_ptr,
    K_ptr,
    V_ptr,
    E_ptr,
    ROWPTR_ptr,  # [N_src+1]
    EDGE_IDS_ptr,  # [M] edge id list grouped by src
    EDGE_DST_ptr,  # [M] dst node for each edge
    D_ptr,  # [N_dst * H] D_j from pass dst-pass
    M_ptr,  # [N_dst * H] saved m_j from fwd
    D_OUT_ptr,  # [N_dst * H * C]
    D_K_ptr,  # [N_src * H * C]
    D_V_ptr,  # [N_src * H * C]
    D_E_ptr,  # [M * H * C]
    N_src,
    H: tl.constexpr,
    C: tl.constexpr,
    out_dtype: tl.constexpr,
):
    src_idx = tl.program_id(0)
    if src_idx >= N_src:
        return

    H_pad: tl.constexpr = triton.next_power_of_2(H)
    C_pad: tl.constexpr = triton.next_power_of_2(C)
    H_mask, H_C_mask, H_C_off = build_masks_and_offsets(H, C, H_pad, C_pad)

    start = tl.load(ROWPTR_ptr + src_idx)
    end = tl.load(ROWPTR_ptr + src_idx + 1)
    num_edges = end - start

    if num_edges == 0:
        zeros = tl.zeros((H_pad * C_pad,), dtype=out_dtype)
        tl.store(D_K_ptr + src_idx * H * C + H_C_off, zeros, mask=H_C_mask)
        tl.store(D_V_ptr + src_idx * H * C + H_C_off, zeros, mask=H_C_mask)
        return

    # Source-side k and v are shared by all edges leaving this source node.
    src_off = src_idx * H * C + H_C_off
    k = tl.load(K_ptr + src_off, mask=H_C_mask, other=0.0).to(tl.float32).reshape((H_pad, C_pad))
    v = tl.load(V_ptr + src_off, mask=H_C_mask, other=0.0).to(tl.float32).reshape((H_pad, C_pad))

    # This pass walks edges grouped by source so one kernel instance can sum
    # dK and dV for one source node and write dE for each edge it touches.
    # The reverse grouping is only an index list. The actual edge tensors are
    # still stored in the original forward / CSC edge order, so those edges
    # are not necessarily contiguous in memory here. `EDGE_IDS_ptr` maps each
    # source-grouped slot back to the real edge id in E_ptr and D_E_ptr.
    accK = tl.zeros((H_pad, C_pad), dtype=tl.float32)
    accV = tl.zeros((H_pad, C_pad), dtype=tl.float32)

    qk_scale: tl.constexpr = 1.0 / tl.sqrt(float(C))

    # for i in tl.range(0, num_edges, warp_specialize=True):
    for i in range(num_edges):
        e_idx = tl.load(EDGE_IDS_ptr + start + i)
        dst = tl.load(EDGE_DST_ptr + e_idx)

        # Load the saved destination-side tensors for this edge.
        dst_off = dst * H * C + H_C_off
        q = tl.load(Q_ptr + dst_off, mask=H_C_mask, other=0.0).to(tl.float32).reshape((H_pad, C_pad))
        out_j = tl.load(OUT_FP32_ptr + dst_off, mask=H_C_mask, other=0.0).to(tl.float32).reshape((H_pad, C_pad))
        d_out = tl.load(D_OUT_ptr + dst_off, mask=H_C_mask, other=0.0).to(tl.float32).reshape((H_pad, C_pad))

        h_off = tl.arange(0, H_pad)
        row_off = dst * H + h_off
        m_j = tl.load(MMAX_ptr + row_off, mask=H_mask, other=float("inf")).to(tl.float32)
        inv_l_j = tl.load(INV_L_ptr + row_off, mask=H_mask, other=0.0).to(tl.float32)

        e_off = e_idx * H * C + H_C_off
        e = tl.load(E_ptr + e_off, mask=H_C_mask, other=0.0).to(tl.float32).reshape((H_pad, C_pad))

        ke = k + e
        ve = v + e

        # Recompute this edge score and softmax probability from the saved
        # forward row stats.
        s_ij = tl.sum(q * ke, axis=-1) * qk_scale
        p_ij = tl.exp(s_ij - m_j) * inv_l_j
        # Use the saved fp32 output here so the small difference (v + e) - out
        # stays accurate in the sharp-softmax cancellation cases.
        centered = tl.sum(d_out * (ve - out_j), axis=-1)
        dS = p_ij * centered

        # dE receives both the value-path and key/score-path contributions.
        dV_edge = p_ij[:, None] * d_out
        dK_edge = dS[:, None] * q * qk_scale
        dE_edge = dV_edge + dK_edge

        tl.store(D_E_ptr + e_off, dE_edge.to(out_dtype).reshape(H_pad * C_pad), mask=H_C_mask)

        accK += dK_edge
        accV += dV_edge

    # Write the final accumulated gradients for this source node.
    tl.store(D_K_ptr + src_off, accK.to(out_dtype).reshape(H_pad * C_pad), mask=H_C_mask)
    tl.store(D_V_ptr + src_off, accV.to(out_dtype).reshape(H_pad * C_pad), mask=H_C_mask)


# TODO(Jan): single bwd pass for non-bipartite graphs


class GraphTransformerFunction(torch.autograd.Function):
    """Two-pass Triton autograd for GraphTransformer.

    Forward walks incoming edges in CSC order, meaning edges are grouped by
    destination node. Backward reuses that same CSC order for dQ, then walks a
    second source-grouped index for dK, dV, and dE.
    """

    def __init__(self):
        if not torch.cuda.is_available():
            raise ValueError(
                "Error. The 'triton' backend was selected for the GraphTransformer but 'torch.cuda.is_available()' returned 'False'. The 'triton' backend is currently only supported on GPUs. To run on other device types, please select a different backend for the GraphTransformer in the models config. If you intend to run on GPUs, please ensure your torch install supports running on GPUs."
            )

    @staticmethod
    def forward(ctx, q, k, v, e, csc, reverse):
        """Args:
        q: [N_dst, H, C]
        k: [N_src, H, C]
        v: [N_src, H, C]
        e: [num_edges, H, C]
        csc: (row, colptr) for destination-major / CSC traversal
             `row` stores the source node per edge, `colptr` stores the start
             and end of each destination row in that edge list.
        reverse: (rowptr, edge_ids, edge_dst) for source-major traversal in backward
                 `edge_ids` points back into the original edge order because the
                 edge tensors themselves stay in that original layout.
        """
        row, colptr = csc
        rowptr, edge_ids, edge_dst = reverse

        # Ensure contiguous memory layout for Triton.
        q, k, v, e = [x.contiguous() for x in (q, k, v, e)]
        row, colptr, rowptr, edge_ids, edge_dst = [x.contiguous() for x in (row, colptr, rowptr, edge_ids, edge_dst)]

        N_dst, H, C = q.shape
        out_saved = torch.empty((N_dst, H, C), device=q.device, dtype=torch.float32)
        m = torch.empty((N_dst, H), device=q.device, dtype=torch.float32)

        # fix: write output in float32 for backward stability, then cast back to input dtype at the end of forward
        _gt_fwd[(N_dst,)](q, k, v, e, m, row, colptr, out_saved, N_dst, H, C, tl.float32)

        # Save tensors for backward
        ctx.save_for_backward(q, k, v, e, out_saved, m, row, colptr, rowptr, edge_ids, edge_dst)
        return out_saved.to(q.dtype)

    @staticmethod
    def backward(ctx, d_out):
        d_out = d_out.contiguous()
        q, k, v, e, out_saved, m, row, colptr, rowptr, edge_ids, edge_dst = ctx.saved_tensors

        N_dst, H, C = q.shape
        N_src = k.shape[0]

        # Infer gradient dtype from incoming gradient tensor
        def torch_dtype_to_triton(dtype):
            if dtype == torch.float16:
                return tl.float16
            elif dtype == torch.bfloat16:
                return tl.bfloat16
            elif dtype == torch.float32:
                return tl.float32
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")

        grad_dtype = torch_dtype_to_triton(d_out.dtype)

        # Allocate gradient outputs.
        dQ = torch.empty_like(q)
        dK = torch.empty_like(k)
        dV = torch.empty_like(v)
        dE = torch.empty_like(e)
        D = torch.empty((N_dst, H), device=q.device, dtype=torch.float32)

        # Pass A: destination nodes (computes D and dQ)
        _gt_bwd_dst_pass[(N_dst,)](q, k, v, e, out_saved, m, row, colptr, d_out, dQ, D, N_dst, H, C, grad_dtype)

        # Pass B: source nodes (accumulate dK, dV, dE)
        _gt_bwd_src_pass[(N_src,)](
            q, k, v, e, rowptr, edge_ids, edge_dst, D, m, d_out, dK, dV, dE, N_src, H, C, grad_dtype
        )

        return dQ, dK, dV, dE, None, None
