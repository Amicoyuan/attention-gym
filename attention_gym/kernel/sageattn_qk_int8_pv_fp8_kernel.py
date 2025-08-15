import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    K_block_ptr,
    V_block_ptr,
    K_scale_ptr,  #
    start_m,
    qk_scale,
    q_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
        K_scale_ptr += lo // BLOCK_N
    # causal = False
    else:
        #
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_scale = tl.load(K_scale_ptr)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        # qk means si
        qk = tl.dot(q, k).to(tl.float32)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale * q_scale * k_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            qk = qk * qk_scale * q_scale * k_scale
            # find current max_ij
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            # si - m_ij
            qk = qk - m_ij[:, None]
        # p~
        p = tl.math.exp2(qk)
        # compute rowsum
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i

        # alpha is li(j-1) The scaling factor
        alpha = tl.math.exp2(m_i - m_ij)
        # rowsum l_i
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]

        # update acc
        v = tl.load(V_block_ptr)

        p = p * 448.0
        fp8_p = p.to(tl.float8e4nv)

        acc += tl.dot(fp8_p, v, out_dtype=tl.float32)
        # update m_i and l_i
        m_i = m_ij
        K_scale_ptr += 1
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    Q_scale,
    K_scale,
    V_scale,
    sm_scale,
    Out,
    Lse,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    qo_len,
    kv_len,
    H: tl.constexpr,
    num_kv_groups: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):

    tl.static_assert(BLOCK_N <= HEAD_DIM)

    # which m block
    start_m = tl.program_id(0)

    # which batch
    off_z = tl.program_id(2)

    # which head_num
    off_h = tl.program_id(1)

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    q_scale_offset = (off_z * H + off_h) * tl.cdiv(qo_len, BLOCK_M)
    k_scale_offset = (off_z * (H // num_kv_groups) + off_h // num_kv_groups) * tl.cdiv(
        kv_len, BLOCK_N
    )
    v_scale_offset = off_z.to(tl.int64) * (H // num_kv_groups * HEAD_DIM) + (
        off_h.to(tl.int64) // num_kv_groups
    ) * (HEAD_DIM)

    k_offset = (
        off_z.to(tl.int64) * stride_kz
        + (off_h // num_kv_groups).to(tl.int64) * stride_kh
    )
    v_offset = (
        off_z.to(tl.int64) * stride_vz
        + (off_h // num_kv_groups).to(tl.int64) * stride_vh
    )
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    # block pointers
    """
        order = (1, 0) means ror-major
        order = (0, 1) means col-major
    """
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(qo_len, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    """
        when v.dtype is fp8 transpose v_layout 
        from [seq_dim, head_dim ] row-major ----->  [seq_dim, head_dim] col-major
    """
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(kv_len, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(0, 1),
    )

    V_scale_ptr = V_scale + v_scale_offset + tl.arange(0, HEAD_DIM)

    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, kv_len),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    K_scale_ptr = K_scale + k_scale_offset

    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(qo_len, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    # rowmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # rowsum
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    # acc buffer
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q_scale = tl.load(Q_scale_ptr)
    v_scale = tl.load(V_scale_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            K_scale_ptr,  #
            start_m,
            qk_scale,
            q_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            kv_len,
            V.dtype.element_ty == tl.float8e4nv,  #
        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            K_scale_ptr,  #
            start_m,
            qk_scale,
            q_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            kv_len,
            V.dtype.element_ty == tl.float8e4nv,  #
        )

    acc = acc / l_i[:, None]
    # dequantize P and V
    acc = acc / 448.0 * v_scale
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

    if RETURN_LSE:
        lse_ptrs = Lse + (off_z * qo_len * H + off_h * qo_len) + offs_m
        l_i = tl.math.log2(l_i) + m_i
        tl.store(lse_ptrs, l_i, mask=(offs_m < qo_len))


def qk_int8_pv_fp8_forward_kernel(
    q,
    k,
    v,
    q_scale,
    k_scale,
    v_scale,
    sm_scale,
    is_causal=False,
    tensor_layout="HND",
    output_dtype=torch.float16,
    return_lse=False,
):
    BLOCK_M = 128
    BLOCK_N = 64

    stage = 3 if is_causal else 1

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q, stride_qk = (
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
        )
        stride_bz_k, stride_h_k, stride_seq_k, stride_kk = (
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
        )
        stride_bz_v, stride_h_v, stride_seq_v, stride_vn = (
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
        )
        stride_bz_o, stride_h_o, stride_seq_o, stride_on = (
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
        )
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q, stride_qk = (
            q.stride(0),
            q.stride(2),
            q.stride(1),
            q.stride(3),
        )
        stride_bz_k, stride_h_k, stride_seq_k, stride_kk = (
            k.stride(0),
            k.stride(2),
            k.stride(1),
            k.stride(3),
        )
        stride_bz_v, stride_h_v, stride_seq_v, stride_vn = (
            v.stride(0),
            v.stride(2),
            v.stride(1),
            v.stride(3),
        )
        stride_bz_o, stride_h_o, stride_seq_o, stride_on = (
            o.stride(0),
            o.stride(2),
            o.stride(1),
            o.stride(3),
        )
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    HEAD_DIM_K = head_dim
    num_kv_groups = h_qo // h_kv

    if return_lse:
        lse = torch.empty([b, h_qo, qo_len], dtype=torch.float32, device=q.device)
    else:
        lse = torch.empty([0], dtype=torch.float32, device="cpu")

    grid = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
    _attn_fwd[grid](
        q,
        k,
        v,
        q_scale,
        k_scale,
        v_scale,
        sm_scale,
        o,
        lse,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_qk,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_kk,
        stride_bz_v,
        stride_h_v,
        stride_seq_v,
        stride_vn,
        stride_bz_o,
        stride_h_o,
        stride_seq_o,
        stride_on,
        qo_len,
        kv_len,
        h_qo,
        num_kv_groups,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM_K,
        STAGE=stage,
        RETURN_LSE=return_lse,
        num_warps=4 if head_dim == 64 else 8,
        num_stages=3 if head_dim == 64 else 4,
    )

    return o, lse
