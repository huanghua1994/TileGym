import cuda.tile as ct
import torch
from typing import Optional
import random
import time
import tilegym
from tilegym.ops.cutile.utils import next_power_of_2

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

@ct.kernel
def group_gemm_kernel_v2(
    A: torch.Tensor,
    B: torch.Tensor,
    M_splits: torch.Tensor,
    M_tile_splits: torch.Tensor,
    group_ids: torch.Tensor,
    C: torch.Tensor,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    M_total: ConstInt,
):
    m_tile_idx = ct.bid(0)
    n_tile_idx = ct.bid(1)
    group_idx = ct.gather(group_ids, m_tile_idx)
    num_k_tiles = ct.num_tiles(A, 1, (TILE_M, TILE_K))
    acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)

    group_M_tile_start_idx = ct.gather(M_tile_splits, group_idx)
    M_tile_idx_in_group = ct.sub(m_tile_idx, group_M_tile_start_idx)
    first_M_index_in_group = ct.gather(M_splits, group_idx)
    last_M_index_in_group = ct.gather(M_splits, ct.add(group_idx, 1))
    M_start = ct.add(first_M_index_in_group, ct.mul(M_tile_idx_in_group, TILE_M))
    if M_start >= M_total:
        return
    N_start = n_tile_idx * TILE_N
    M_indices = M_start + ct.arange(TILE_M, dtype=torch.int32)
    N_indices = N_start + ct.arange(TILE_N, dtype=torch.int32)

    for k_tile_idx in range(num_k_tiles):
        K_indices = k_tile_idx * TILE_K + ct.arange(TILE_K, dtype=torch.int32)
        A_tile = ct.gather(A, (M_indices[:, None], K_indices[None, :]))
        B_tile = ct.load(B, index=(group_idx, k_tile_idx, n_tile_idx), shape=(1, TILE_K, TILE_N))
        B_tile = ct.reshape(B_tile, (TILE_K, TILE_N))
        acc = ct.mma(A_tile, B_tile, acc)
    acc = ct.astype(acc, C.dtype)

    # Use out-of-bound indices to skip storing
    store_mask = (M_indices < last_M_index_in_group)
    store_M_indices = ct.where(store_mask, M_indices, M_total+1)
    ct.scatter(C, (store_M_indices[:, None], N_indices[None, :]), acc)


def group_gemm_v2(
    A: torch.Tensor,
    B: torch.Tensor,
    M_splits: torch.Tensor,
    C: Optional[torch.Tensor] = None,
    *,
    TILE_M: int = 128,
    TILE_N: int = 128,
    TILE_K: int = 64,
) -> Optional[torch.Tensor]:
    """
    Group GEMM

    Args:
        A: Shape (M, K), list of A matrices concatenated along the first dimension
        B: Shape (G, K, N), list of B matrices
        M_splits: Shape (G+1,), list of integers or tensor of integers, the split
                  points of M dimension
        TILE_M: Tile size for the M dimension
        TILE_N: Tile size for the N dimension
        TILE_K: Tile size for the K dimension

    Returns:
        C: Shape (M, N), list of C matrices concatenated along the first dimension, where
           C[M_splits[i]:M_splits[i+1], :] = A[M_splits[i]:M_splits[i+1], :] * B[i, :, :]
    """
    device = A.device
    dtype = A.dtype
    M, _K = A.shape
    G, _K, N = B.shape
    assert M_splits.shape[0] == G + 1

    def align_tile_size(tile_size: int, tile_size_name: str) -> int:
        new_tile_size = next_power_of_2(tile_size)
        if new_tile_size != tile_size:
            print(f"Tile size for {tile_size_name} is not a power of 2, aligning to {new_tile_size}")
        return new_tile_size
    
    TILE_M = align_tile_size(TILE_M, "TILE_M")
    TILE_N = align_tile_size(TILE_N, "TILE_N")
    TILE_K = align_tile_size(TILE_K, "TILE_K")

    empty_input_C = C is None
    if empty_input_C:
        C = torch.empty((M, N), device=device, dtype=dtype)

    M_list = M_splits[1:] - M_splits[:-1]
    num_M_tiles = ct.cdiv(M_list, TILE_M)
    M_tile_splits = torch.cat([
        torch.zeros(1, device=device, dtype=torch.int32),
        torch.cumsum(num_M_tiles, dim=0)
    ])
    group_ids = torch.repeat_interleave(
        torch.arange(G, device=device),
        num_M_tiles
    )
    num_M_tiles_upper_bound = (M + TILE_M - 1) // TILE_M + G
    total_N_tiles = ct.cdiv(N, TILE_N)

    grid = (num_M_tiles_upper_bound, total_N_tiles)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        group_gemm_kernel_v2,
        (
            A,
            B,
            M_splits,
            M_tile_splits,
            group_ids,
            C,
            TILE_M,
            TILE_N,
            TILE_K,
            M,
        ),
    )

    if empty_input_C:
        return C


@ct.kernel
def group_gemm_kernel_v3(
    A: torch.Tensor,
    B: torch.Tensor,
    M_splits: torch.Tensor,
    M_tile_splits: torch.Tensor,
    group_ids: torch.Tensor,
    C: torch.Tensor,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    M_total: ConstInt,
    num_sms: ConstInt,
    num_n_tiles: ConstInt,
    num_tiles_upper_bound: ConstInt,
):
    tile_idx = ct.bid(0)
    
    for curr_tile_idx in range(tile_idx, num_tiles_upper_bound, num_sms):
        m_tile_idx = curr_tile_idx // num_n_tiles
        n_tile_idx = curr_tile_idx % num_n_tiles
        group_idx = ct.gather(group_ids, m_tile_idx)
        num_k_tiles = ct.num_tiles(A, 1, (TILE_M, TILE_K))
        acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)

        group_M_tile_start_idx = ct.gather(M_tile_splits, group_idx)
        M_tile_idx_in_group = ct.sub(m_tile_idx, group_M_tile_start_idx)
        first_M_index_in_group = ct.gather(M_splits, group_idx)
        M_start = ct.add(first_M_index_in_group, ct.mul(M_tile_idx_in_group, TILE_M))
        if M_start >= M_total:
            continue
        N_start = n_tile_idx * TILE_N
        M_indices = M_start + ct.arange(TILE_M, dtype=torch.int32)
        N_indices = N_start + ct.arange(TILE_N, dtype=torch.int32)

        for k_tile_idx in range(num_k_tiles):
            K_indices = k_tile_idx * TILE_K + ct.arange(TILE_K, dtype=torch.int32)
            A_tile = ct.gather(A, (M_indices[:, None], K_indices[None, :]))
            B_tile = ct.load(B, index=(group_idx, k_tile_idx, n_tile_idx), shape=(1, TILE_K, TILE_N))
            B_tile = ct.reshape(B_tile, (TILE_K, TILE_N))
            acc = ct.mma(A_tile, B_tile, acc)
        acc = ct.astype(acc, C.dtype)

        # Use out-of-bound indices to skip storing
        last_M_index_in_group = ct.gather(M_splits, ct.add(group_idx, 1))
        store_mask = (M_indices < last_M_index_in_group)
        store_M_indices = ct.where(store_mask, M_indices, M_total+1)
        ct.scatter(C, (store_M_indices[:, None], N_indices[None, :]), acc)


def group_gemm_v3(
    A: torch.Tensor,
    B: torch.Tensor,
    M_splits: torch.Tensor,
    C: Optional[torch.Tensor] = None,
    *,
    TILE_M: int = 128,
    TILE_N: int = 128,
    TILE_K: int = 64,
) -> Optional[torch.Tensor]:
    """
    Group GEMM

    Args:
        A: Shape (M, K), list of A matrices concatenated along the first dimension
        B: Shape (G, K, N), list of B matrices
        M_splits: Shape (G+1,), list of integers or tensor of integers, the split
                  points of M dimension
        TILE_M: Tile size for the M dimension
        TILE_N: Tile size for the N dimension
        TILE_K: Tile size for the K dimension

    Returns:
        C: Shape (M, N), list of C matrices concatenated along the first dimension, where
           C[M_splits[i]:M_splits[i+1], :] = A[M_splits[i]:M_splits[i+1], :] * B[i, :, :]
    """
    device = A.device
    dtype = A.dtype
    M, _K = A.shape
    G, _K, N = B.shape
    assert M_splits.shape[0] == G + 1

    def align_tile_size(tile_size: int, tile_size_name: str) -> int:
        new_tile_size = next_power_of_2(tile_size)
        if new_tile_size != tile_size:
            print(f"Tile size for {tile_size_name} is not a power of 2, aligning to {new_tile_size}")
        return new_tile_size
    
    TILE_M = align_tile_size(TILE_M, "TILE_M")
    TILE_N = align_tile_size(TILE_N, "TILE_N")
    TILE_K = align_tile_size(TILE_K, "TILE_K")

    empty_input_C = C is None
    if empty_input_C:
        C = torch.empty((M, N), device=device, dtype=dtype)

    M_list = M_splits[1:] - M_splits[:-1]
    num_M_tiles = ct.cdiv(M_list, TILE_M)
    M_tile_splits = torch.cat([
        torch.zeros(1, device=device, dtype=torch.int32),
        torch.cumsum(num_M_tiles, dim=0)
    ])
    group_ids = torch.repeat_interleave(
        torch.arange(G, device=device),
        num_M_tiles
    )
    num_compact_m_tiles = (M + TILE_M - 1) // TILE_M
    num_m_tiles_upper_bound = num_compact_m_tiles + G
    num_n_tiles = (N + TILE_N - 1) // TILE_N
    num_tiles_upper_bound = num_m_tiles_upper_bound * num_n_tiles

    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    grid = (NUM_SMS, )
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        group_gemm_kernel_v3,
        (
            A,
            B,
            M_splits,
            M_tile_splits,
            group_ids,
            C,
            TILE_M,
            TILE_N,
            TILE_K,
            M,
            NUM_SMS,
            num_n_tiles,
            num_tiles_upper_bound,
        ),
    )

    if empty_input_C:
        return C


def bench_group_gemm(
    num_groups: int,
    M_total: int,
    N: int,
    K: int,
    dtype: torch.dtype = torch.float16,
    distribution: str = "nearly_balanced",
    num_warmup: int = 5,
    num_test: int = 20,
):
    M_i = M_total // num_groups
    if distribution == "balanced":
        M_list = [M_i for _ in range(num_groups-1)]
        M_list.append(M_total - sum(M_list))
    elif distribution == "nearly_balanced":
        var_range = M_i // 10
        max_M = M_i + var_range
        min_M = M_i - var_range
        M_list = [random.randint(min_M, max_M) for _ in range(num_groups-1)]
        last_M = max(0, M_total - sum(M_list))
        M_list.append(last_M)
    elif distribution == "random":
        var_range = int(M_i * 0.8)
        max_M = M_i + var_range
        min_M = M_i - var_range
        M_list = [random.randint(min_M, max_M) for _ in range(num_groups-1)]
        last_M = max(0, M_total - sum(M_list))
        M_list.append(last_M)
    else:
        raise ValueError(f"Invalid M size distribution: {distribution}")

    group_A = []
    group_B = []
    group_C_ref = []
    for i in range(num_groups):
        A_i = torch.randn((M_list[i], K), device="cuda", dtype=dtype)
        B_i = torch.randn((K, N), device="cuda", dtype=dtype)
        C_i_ref = torch.matmul(A_i, B_i)
        group_A.append(A_i)
        group_B.append(B_i)
        group_C_ref.append(C_i_ref)

    A = torch.cat(group_A, dim=0)
    B = torch.cat(group_B, dim=0).reshape(num_groups, K, N)
    M_list_dev = torch.tensor(M_list, device="cuda", dtype=torch.int32)
    M_splits_dev = torch.cat([
        torch.zeros(1, device="cuda", dtype=torch.int32),
        torch.cumsum(M_list_dev, dim=0),
    ])

    for _ in range(num_warmup):
        group_C_v1 = tilegym.ops.group_gemm(group_A, group_B)
        C_v2 = group_gemm_v2(A, B, M_splits_dev)
        C_v3 = group_gemm_v3(A, B, M_splits_dev)

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_test):
        group_C_v1 = tilegym.ops.group_gemm(group_A, group_B)
    torch.cuda.synchronize()
    v1_time = (time.time() - start_time) / num_test
    v1_tflops = 2 * M_total * N * K / v1_time / 1e12

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_test):
        C_v2 = group_gemm_v2(A, B, M_splits_dev)
    torch.cuda.synchronize()
    v2_time = (time.time() - start_time) / num_test
    v2_tflops = 2 * M_total * N * K / v2_time / 1e12

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_test):
        C_v3 = group_gemm_v3(A, B, M_splits_dev)
    torch.cuda.synchronize()
    v3_time = (time.time() - start_time) / num_test
    v3_tflops = 2 * M_total * N * K / v3_time / 1e12

    M_splits_host = M_splits_dev.cpu().numpy()
    for i in range(num_groups):
        C_i_ref = group_C_ref[i]
        C_i_v1 = group_C_v1[i]
        C_i_v2 = C_v2[M_splits_host[i]:M_splits_host[i+1], :]
        C_i_v3 = C_v3[M_splits_host[i]:M_splits_host[i+1], :]
        assert C_i_v1.shape == C_i_v2.shape, f"Output matrix {i} shape mismatch: {C_i_v1.shape} != {C_i_v2.shape}"
        assert C_i_v1.shape == C_i_v3.shape, f"Output matrix {i} shape mismatch: {C_i_v1.shape} != {C_i_v3.shape}"
        if C_i_v1.numel() == 0:
            continue
        assert torch.allclose(C_i_ref, C_i_v1, atol=1e-2, rtol=1e-2)
        assert torch.allclose(C_i_ref, C_i_v2, atol=1e-2, rtol=1e-2)
        assert torch.allclose(C_i_ref, C_i_v3, atol=1e-2, rtol=1e-2)

    return v1_tflops, v2_tflops, v3_tflops


if __name__ == "__main__":
    torch.manual_seed(0)
    dtype = torch.float16
    num_tokens_list = [1, 32, 256, 1024, 4096, 8192]
    # DSv3 MoE parameters
    num_groups = 320  # num_experts
    topk = 8
    K = 7168          # hidden_size
    N = 2048          # moe_intermediate_size
    print(f"num_groups: {num_groups}, topk: {topk}, K: {K}, N: {N}")
    print(f"M_total, ref TFLOPS, v2 TFLOPS, v3 TFLOPS")
    for num_tokens in num_tokens_list:
        M_total = num_tokens * topk
        v1_tflops, v2_tflops, v3_tflops = bench_group_gemm(num_groups, M_total, N, K, dtype)
        print(f"{M_total}, {v1_tflops:.2f}, {v2_tflops:.2f}, {v3_tflops:.2f}")
