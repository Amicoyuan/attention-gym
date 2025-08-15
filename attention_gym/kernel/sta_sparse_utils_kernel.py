import torch
import triton
import triton.language as tl


@triton.jit
def get_tile_t_x_y(
    idx: int, total_tile_size: int, canvas_tile_h: int, canvas_tile_w: int
):
    tile_id = idx // total_tile_size
    tile_t = tile_id // (canvas_tile_h * canvas_tile_w)
    tile_h = (tile_id % (canvas_tile_h * canvas_tile_w)) // canvas_tile_w
    tile_w = tile_id % canvas_tile_w
    return tile_t, tile_h, tile_w


@triton.jit
def sta_mask_kernel(
    block_mask,
    qo_len: tl.constexpr,
    kv_len: tl.constexpr,
    canvas_t: tl.constexpr,
    canvas_h: tl.constexpr,
    canvas_w: tl.constexpr,
    kernel_t: tl.constexpr,
    kernel_h: tl.constexpr,
    kernel_w: tl.constexpr,
    tile_t_size: tl.constexpr,
    tile_h_size: tl.constexpr,
    tile_w_size: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    img_seq_len: tl.constexpr,
    text_length: tl.constexpr,
    total_tile_size: tl.constexpr,
    canvas_tile_t: tl.constexpr,
    canvas_tile_h: tl.constexpr,
    canvas_tile_w: tl.constexpr,
):
    qo_blk_id = tl.program_id(0)
    kv_blk_id = tl.program_id(1)

    offs_qo = qo_blk_id
    offs_kv = kv_blk_id

    qo_id = offs_qo * block_m
    kv_id = offs_kv * block_n
    block_mask_ptr = block_mask + offs_qo * (kv_len // block_n) + kv_blk_id
    q_t_tile, q_x_tile, q_y_tile = get_tile_t_x_y(
        qo_id, total_tile_size, canvas_tile_h, canvas_tile_w
    )
    kv_t_tile, kv_x_tile, kv_y_tile = get_tile_t_x_y(
        kv_id, total_tile_size, canvas_tile_h, canvas_tile_w
    )
    kernel_center_t = max(
        kernel_t // 2, min(q_t_tile, (canvas_tile_t - 1) - kernel_t // 2)
    )
    kernel_center_x = max(
        kernel_h // 2, min(q_x_tile, (canvas_tile_h - 1) - kernel_h // 2)
    )
    kernel_center_y = max(
        kernel_w // 2, min(q_y_tile, (canvas_tile_w - 1) - kernel_w // 2)
    )
    time_mask = tl.abs(kernel_center_t - kv_t_tile) <= kernel_t // 2
    hori_mask = tl.abs(kernel_center_x - kv_x_tile) <= kernel_h // 2
    vert_mask = tl.abs(kernel_center_y - kv_y_tile) <= kernel_w // 2
    image_mask = (qo_id < img_seq_len) & (kv_id < img_seq_len)
    image_to_text_mask = (
        (qo_id < img_seq_len)
        & (kv_id >= img_seq_len)
        & (kv_id < img_seq_len + text_length)
    )
    text_to_all_mask = (qo_id >= img_seq_len) & (kv_id < img_seq_len + text_length)
    mask_mode = (
        (image_mask & time_mask & hori_mask & vert_mask)
        | image_to_text_mask
        | text_to_all_mask
    )
    tl.store(block_mask_ptr, mask_mode)


def get_sliding_tile_attention_mask_triton(
    qo_len,
    kv_len,
    kernel_size,
    tile_size,
    img_size,
    text_length,
    device,
    text_max_len=256,
):

    block_m, block_n = 128, 64
    block_mask = torch.zeros(
        (1, 1, qo_len // block_m, kv_len // block_n), dtype=torch.bool, device=device
    )

    img_seq_len = img_size[0] * img_size[1] * img_size[2]
    canvas_t, canvas_h, canvas_w = img_size
    kernel_t, kernel_h, kernel_w = kernel_size
    tile_t_size, tile_h_size, tile_w_size = tile_size
    total_tile_size = tile_t_size * tile_h_size * tile_w_size
    canvas_tile_t, canvas_tile_h, canvas_tile_w = (
        canvas_t // tile_t_size,
        canvas_h // tile_h_size,
        canvas_w // tile_w_size,
    )
    img_seq_len = canvas_t * canvas_h * canvas_w

    block_m_num = qo_len // block_m
    block_n_num = kv_len // block_n
    grid = (block_m_num, block_n_num, 1)
    sta_mask_kernel[grid](
        block_mask,
        qo_len,
        kv_len,
        canvas_t,
        canvas_h,
        canvas_w,
        kernel_t,
        kernel_h,
        kernel_w,
        tile_t_size,
        tile_h_size,
        tile_w_size,
        block_m,
        block_n,
        img_seq_len,
        text_length,
        total_tile_size,
        canvas_tile_t,
        canvas_tile_h,
        canvas_tile_w,
    )

    return block_mask
