import numpy as np
from typing import Optional, Tuple

# -----------------------------------------------------------
# Config
# -----------------------------------------------------------
N = 256
D_Head = 64

# tiles for the proposed HW-style kernel
T_M = 8
T_O = 8
T_N = 8
T_K = 8

print(f"\nT_M: {T_M}, T_O: {T_O}, T_N: {T_N}, T_K: {T_K}")

np.set_printoptions(precision=6, suppress=True, linewidth=120)

# -----------------------------------------------------------
# Data on off-chip memory
# -----------------------------------------------------------

# query_head: N * D_Head
print("\n--- query_head ---")
query_head = np.random.randn(N, D_Head).astype(np.float32)
print(f"Shape: {query_head.shape}")

# key_head: N * D_Head
print("--- key_head ---")
key_head = np.random.randn(N, D_Head).astype(np.float32)
print(f"Shape: {key_head.shape}")

# value_head: N * D_Head
print("--- value_head ---")
value_head = np.random.randn(N, D_Head).astype(np.float32)
print(f"Shape: {value_head.shape}")

# datty_head: N * D_Head
print("--- datty_head ---")
datty_head = np.random.randn(N, D_Head).astype(np.float32)
print(f"Shape: {datty_head.shape}")

# att_mask: N * N
print("--- att_mask ---")
att_mask = np.tril(np.ones((N, N), dtype=np.float32))
print(f"Shape: {att_mask.shape}")

# Matrices to be stored in Off-chip
atty_prop = np.zeros((N, D_Head), dtype=np.float32) # Off-chip output 'atty' (O)
att_prop = np.zeros((N, N), dtype=np.float32)       # Off-chip output 'att'  (P)

# -----------------------------------------------------------
# Profiling helpers
# -----------------------------------------------------------
class Prof:
    def __init__(self, name):
        self.name = name
        self.offchip_load_bytes = 0
        self.offchip_store_bytes = 0
        self.onchip_current = 0
        self.onchip_peak = 0

    def _bump_peak(self):
        if self.onchip_current > self.onchip_peak:
            self.onchip_peak = self.onchip_current

    def alloc_onchip(self, shape, dtype=np.float32, zero=True):
        """Allocate an on-chip buffer and track peak usage."""
        arr = np.zeros(shape, dtype=dtype) if zero else np.empty(shape, dtype=dtype)
        self.onchip_current += arr.nbytes
        self._bump_peak()
        return arr

    def free_onchip(self, arr):
        """Free an on-chip buffer (tracking only; Python GC actually frees)."""
        self.onchip_current -= arr.nbytes
        if self.onchip_current < 0:
            self.onchip_current = 0  # safety

    def load_from_offchip(self, nbytes):
        """Account for off-chip -> on-chip loads."""
        self.offchip_load_bytes += nbytes

    def store_to_offchip(self, nbytes):
        """Account for on-chip -> off-chip writes."""
        self.offchip_store_bytes += nbytes

# -----------------------------------------------------------
# Unit Adjusted human_bytes function (Required for MB/KB output)
# -----------------------------------------------------------
def human_bytes(B, target_unit=None):
    """
    Converts bytes to human-readable format, targeting MB for Off-chip (store/load) 
    and KB for On-chip (peak).
    """
    if target_unit == 'MB':
        return f"{B / (1024**2):.3f}"
    elif target_unit == 'KB':
        return f"{B / 1024:.3f}"
    
    # Default fallback
    if B < 1024: return f"{B}"
    KB = B / 1024
    if KB < 1024: return f"{KB:.3f}"
    MB = KB / 1024
    return f"{MB:.3f}"

# -----------------------------------------------------------
# Method A FWD: (Fully Tiled, O(1) Peak Memory)
# -----------------------------------------------------------
def compute_fwd_proposed():
    # Initialize profiler for this function
    prof = Prof("fwd_standard_fused_O1")
    flops = 0

    # Get pointers to the global Off-chip arrays for storing results
    global atty_prop, att_prop

    # Load scalar options from global scope
    tau          = float(globals().get("tau", 1.0))
    has_mask     = ("att_mask" in globals()) and (att_mask is not None)
    p_drop       = float(globals().get("p_drop", 0.0))
    dropout_seed = int(globals().get("dropout_seed", 12345))

    def make_dropout_tile(T_M, T_O, i0, j0):
        # Generates a dropout mask Z for a specific tile (i0, j0)
        # This function is deterministic based on the seed and tile indices.
        if p_drop <= 0.0:
            return None
        # Use a deterministic random seed based on tile indices
        rs = np.random.RandomState((dropout_seed ^ (i0 * 1000003) ^ (j0 * 2654435761)) & 0x7FFFFFFF)
        # Allocate the Z tile on-chip
        Z = prof.alloc_onchip((T_M, T_O), np.float32)
        keep, scale = 1.0 - p_drop, 1.0 / (1.0 - p_drop) # Scale factor for inverted dropout
        for t1 in range(T_M):
            for t2 in range(T_O):
                Z[t1, t2] = scale if (rs.rand() >= p_drop) else 0.0
        return Z

    print("--- Running Profiled Forward Pass (Standard, Fused, Tiled) ---")

    # ---------- Fused Kernel: Compute O, P ----------
    
    # Outer loop: Iterate over row-blocks (tiles) of Q, O, P (dimension N -> T_M)
    for i0 in range(0, N, T_M):

        # (0) Online-softmax running state for this i-tile (row-wise)
        # These buffers are O(T_M), independent of N and D_Head.
        
        # m_tile: Running maximum value for each row in this T_M block
        m_tile = prof.alloc_onchip((T_M,), np.float32);  m_tile.fill(-np.inf)
        # l_tile: Running denominator (sum of exps) for each row in this T_M block
        l_tile = prof.alloc_onchip((T_M,), np.float32)   # zeros

        # Inner loop: Iterate over column-blocks (tiles) of K, V, P (dimension N -> T_O)
        # This loop streams K and V to compute the final O and P for the current row-block.
        for j0 in range(0, N, T_O):

            # (1) Build S = τ * (Q K^T) on T_M×T_O tile, accumulating over D_Head subtiles
            # This block achieves O(1) memory by tiling D_Head into T_K chunks.
            S_blk = prof.alloc_onchip((T_M, T_O), np.float32)  # zero-inited
            
            # k0 loop: Tile the D_Head dimension into T_K chunks
            for k0 in range(0, D_Head, T_K):
                
                # Q_sub load (T_M x T_K) - On-chip buffer is D_Head-independent
                Q_sub = prof.alloc_onchip((T_M, T_K), np.float32)
                for t1 in range(T_M):
                    # Off-chip read for Q-subtile
                    Q_sub[t1, :T_K] = query_head[i0 + t1, k0:k0 + T_K]
                prof.load_from_offchip(Q_sub.nbytes) # Account for Off-chip read

                # K_sub load (T_O x T_K) - On-chip buffer is D_Head-independent
                K_sub = prof.alloc_onchip((T_O, T_K), np.float32)
                for t2 in range(T_O):
                    K_sub[t2, :T_K] = key_head[j0 + t2, k0:k0 + T_K]
                prof.load_from_offchip(K_sub.nbytes) # Account for Off-chip read

                # S_blk += tau * Q_sub @ K_sub^T
                # Accumulate the S_blk result on-chip
                for t1 in range(T_M):
                    for t2 in range(T_O):
                        acc = 0.0
                        for k in range(T_K):
                            acc += Q_sub[t1, k] * K_sub[t2, k];  flops += 2  # mul + add
                        S_blk[t1, t2] += tau * acc;              flops += 2  # mul (·τ) + add

                # Free the T_K chunks immediately after use
                prof.free_onchip(K_sub); prof.free_onchip(Q_sub)

            # (2) Mask (optional): set masked logits to -inf
            if has_masK := has_mask: # Python 3.8+ walrus operator
                M_blk = prof.alloc_onchip((T_M, T_O), np.float32)
                for t1 in range(T_M):
                    M_blk[t1, :T_O] = att_mask[i0 + t1, j0:j0 + T_O]
                prof.load_from_offchip(M_blk.nbytes)
                for t1 in range(T_M):
                    for t2 in range(T_O):
                        if not np.isfinite(S_blk[t1, t2]) or M_blk[t1, t2] <= 0.0:
                            S_blk[t1, t2] = float("-inf") # Apply mask
                prof.free_onchip(M_blk)

            # (3) Online softmax per row: compute E = exp(S - rowmax), sumexp; update (m, l)
            
            # Find the local max of the current S_blk
            rowmax = prof.alloc_onchip((T_M,), np.float32)
            for t1 in range(T_M):
                mx = -np.inf if not np.isfinite(m_tile[t1]) else m_tile[t1]
                for t2 in range(T_O):
                    if S_blk[t1, t2] > mx:
                        mx = S_blk[t1, t2]
                if not np.isfinite(mx):
                    mx = 0.0  # fully masked row guard (prevents nan)
                rowmax[t1] = mx

            # E_blk = exp(S_blk - m_local) (Local Numerator)
            E_blk  = prof.alloc_onchip((T_M, T_O), np.float32)
            # sumexp = sum(E_blk) (Local Denominator Sum)
            sumexp = prof.alloc_onchip((T_M,),     np.float32)  # zeros
            for t1 in range(T_M):
                sh = rowmax[t1];  se = 0.0
                for t2 in range(T_O):
                    e = np.exp(S_blk[t1, t2] - sh)   # exp() is not counted in FLOPs by policy
                    E_blk[t1, t2] = e
                    se += e;                          flops += 1  # add
                sumexp[t1] = se

            # Core Online Softmax statistics update
            l_hat  = prof.alloc_onchip((T_M,), np.float32) # l_old_rescaled
            l_new  = prof.alloc_onchip((T_M,), np.float32) # l_new
            invnew = prof.alloc_onchip((T_M,), np.float32) # 1.0 / l_new1
            for t1 in range(T_M):
                # Rescaling factor for old statistics
                rfac = np.exp(m_tile[t1] - rowmax[t1]) if np.isfinite(m_tile[t1]) else 0.0
                l_hat[t1] = l_tile[t1] * rfac;                            flops += 1  # mul
                l_new[t1] = l_hat[t1] + sumexp[t1];                       flops += 1  # add
                invnew[t1] = 1.0 / (l_new[t1] if l_new[t1] > 0 else 1.0)  # division not counted

            # (4) P_tile (clean) and store to Off-chip
            # P_blk = E_blk / l_new
            P_blk = prof.alloc_onchip((T_M, T_O), np.float32)
            for t1 in range(T_M):
                inv_l = invnew[t1]
                for t2 in range(T_O):
                    P_blk[t1, t2] = E_blk[t1, t2] * inv_l;  flops += 1  # mul
            
            # Store the final P_clean tile to Off-chip for the backward pass
            att_prop[i0:i0 + T_M, j0:j0 + T_O] = P_blk
            prof.store_to_offchip(P_blk.nbytes)
            prof.free_onchip(P_blk)

            # (5) Update O in channel sub-tiles using P_drop = P_clean ⊙ Z
            # This block achieves O(1) memory by tiling D_Head into T_N chunks
            # and performing a Read-Modify-Write (RMW) on atty_prop in Off-chip.
            
            Z_blk = make_dropout_tile(T_M, T_O, i0, j0)
            
            # k0 loop: Tile the D_Head dimension into T_N chunks
            for k0 in range(0, D_Head, T_N):

                # Load O_chunk (T_M x T_N) from Off-chip
                O_chunk = prof.alloc_onchip((T_M, T_N), np.float32)
                for t1 in range(T_M):
                    O_chunk[t1, :T_N] = atty_prop[i0 + t1, k0:k0 + T_N]
                prof.load_from_offchip(O_chunk.nbytes) # RMW Read

                # Rescale O_chunk by l_hat/l_new
                # This is the first part of the stable online softmax update: O_old * l_old_scaled
                for t1 in range(T_M):
                    alpha = l_hat[t1] * invnew[t1];              flops += 1  # mul
                    for kk in range(T_N):
                        O_chunk[t1, kk] *= alpha;                flops += 1  # mul
                
                # V_sub load (T_O x T_N) - D_Head independent
                V_sub = prof.alloc_onchip((T_O, T_N), np.float32)
                for t2 in range(T_O):
                    V_sub[t2, :T_N] = value_head[j0 + t2, k0:k0 + T_N]
                prof.load_from_offchip(V_sub.nbytes)

                # Compute O_chunk += (E (·Z)) @ V_sub ; then *invnew
                # This is the second part: (O_old_scaled + O_new) / l_new
                
                # acc_row: on-chip scratchpad for (E*Z) @ V_sub
                acc_row = prof.alloc_onchip((T_N,), np.float32)
                for t1 in range(T_M):
                    acc_row.fill(0.0) # Reset scratchpad
                    
                    # Compute acc_row = (E*Z) @ V_sub
                    for t2 in range(T_O):
                        g = E_blk[t1, t2]
                        if Z_blk is not None:
                            g *= Z_blk[t1, t2]                    # mul (not in FLOPs by policy)
                        for kk in range(T_N):
                            acc_row[kk] += g * V_sub[t2, kk];     flops += 2  # mul+add
                    
                    # Combine and apply final normalization
                    beta = invnew[t1]                              # 1.0 / l_new
                    for kk in range(T_N):
                        O_chunk[t1, kk] += beta * acc_row[kk];    flops += 2  # mul+add
                
                prof.free_onchip(acc_row)
                prof.free_onchip(V_sub)

                # Store updated O_chunk back to Off-chip
                atty_prop[i0:i0 + T_M, k0:k0 + T_N] = O_chunk
                prof.store_to_offchip(O_chunk.nbytes) # RMW Write
                prof.free_onchip(O_chunk)

            if Z_blk is not None:
                prof.free_onchip(Z_blk)

            # (6) Finalize row-wise state (m, l) for the next j-tile
            for t1 in range(T_M):
                m_tile[t1] = max(m_tile[t1], rowmax[t1]) # Update m with the local max
                l_tile[t1] = l_new[t1]                  # Update l with the new denominator

            # Free buffers used in j0 loop
            prof.free_onchip(invnew);  prof.free_onchip(l_new); prof.free_onchip(l_hat)
            prof.free_onchip(sumexp);  prof.free_onchip(E_blk)
            prof.free_onchip(rowmax);  prof.free_onchip(S_blk)

        # Free buffers used in i0 loop
        prof.free_onchip(m_tile); prof.free_onchip(l_tile)

    metrics = {
        "offchip_load":        prof.offchip_load_bytes,
        "offchip_store":       prof.offchip_store_bytes,
        "offchip_load_h":      human_bytes(prof.offchip_load_bytes, 'MB'), # Use human_bytes with MB target
        "offchip_store_h":     human_bytes(prof.offchip_store_bytes, 'MB'), # Use human_bytes with MB target
        "onchip_peak_bytes":   prof.onchip_peak, # Corrected: Access the attribute
        "onchip_peak_h":       human_bytes(prof.onchip_peak, 'KB'), # Use human_bytes with KB target
        "flops":               flops
    }
    return atty_prop, att_prop, metrics

# -----------------------------------------------------------
# Method A BWD: (Loads att(P) AND atty(O) from Off-chip)
# -----------------------------------------------------------
def compute_bwd_proposed(atty, att):
    # 'atty' is the O matrix (N x D_Head) from Off-chip
    # 'att' is the P matrix (N x N) from Off-chip
    prof = Prof("proposed_bwd")
    flops = 0

    # Initialize Off-chip-sized output gradient buffers
    d_query_head_proposed = np.zeros((N, D_Head), dtype=np.float32)
    d_value_head_proposed = np.zeros((N, D_Head), dtype=np.float32)
    d_key_head_proposed   = np.zeros((N, D_Head), dtype=np.float32)

    # Load scalar options
    p_drop       = float(globals().get("p_drop", 0.0))
    dropout_seed = int(globals().get("dropout_seed", 12345))
    tau          = float(globals().get("tau", 1.0)) # Scale factor for dQ, dK

    def _make_dropout_Z_tile(T_M, T_O, i0, j0):
        # Deterministically regenerates the dropout mask Z for a specific tile (i0, j0)
        if p_drop <= 0.0: return None
        rs = np.random.RandomState((dropout_seed ^ (i0 * 1000003) ^ (j0 * 2654435761)) & 0x7FFFFFFF)
        Z = prof.alloc_onchip((T_M, T_O), np.float32)
        keep = 1.0 - p_drop; scale = 1.0 / keep
        rnd = rs.rand(T_M, T_O)
        for t1 in range(T_M):
            for t2 in range(T_O):
                Z[t1, t2] = scale if (rnd[t1, t2] >= p_drop) else 0.0
        return Z
    
    print("--- Running Profiled Backward Pass (Standard, Fused, Tiled) ---")

    # Outer loop: Iterate over row-blocks (tiles) of Q/O/D
    for att_i in range(N // T_M):
        i0 = att_i * T_M

        # --- Pass 0: Compute D_i = rowsum(dO_i ⊙ O_i) ---
        # D_i is a (T_M,) vector needed for the dS calculation in Pass 1.
        # This pass is tiled over D_Head to maintain O(1) on-chip memory.
        
        on_chip_D = prof.alloc_onchip((T_M,), np.float32)  # zeros
        
        # k0 loop: Tile the D_Head dimension into T_K chunks
        for k0 in range(0, D_Head, T_K):
            # Load dO chunk
            dO_blk = prof.alloc_onchip((T_M, T_K), np.float32)
            # Load O (atty) chunk
            O_blk  = prof.alloc_onchip((T_M, T_K), np.float32)
            for t1 in range(T_M):
                for k in range(T_K):
                    dO_blk[t1, k] = datty_head[i0 + t1, k0 + k]
                    O_blk[t1, k]  = atty[i0 + t1, k0 + k]
            prof.load_from_offchip(dO_blk.nbytes + O_blk.nbytes) # Account for Off-chip read
            
            # Compute (dO_blk ⊙ O_blk) and accumulate into D_i
            for t1 in range(T_M):
                acc = 0.0
                for k in range(T_K):
                    acc += dO_blk[t1, k] * O_blk[t1, k]; flops += 2 # mul+add
                on_chip_D[t1] += acc; flops += 1 # accumulate
            prof.free_onchip(O_blk); prof.free_onchip(dO_blk)
        # --- End of Pass 0 ---

        
        # --- Pass 1: Compute dV, dP, dS, dQ, dK ---
        # Inner loop: Iterate over column-blocks (tiles) of P, K, V
        for q_h_i in range(N // T_O):
            j0 = q_h_i * T_O

            # (A) Load P_clean tile from Off-chip
            P_clean = prof.alloc_onchip((T_M, T_O), np.float32)
            for t1 in range(T_M):
                for t2 in range(T_O):
                    P_clean[t1, t2] = att[i0 + t1, j0 + t2]
            prof.load_from_offchip(P_clean.nbytes) # Load att(P) tile

            # (B) Create P_drop = P_clean ⊙ Z
            P_drop = prof.alloc_onchip((T_M, T_O), np.float32)
            for t1 in range(T_M):
                for t2 in range(T_O):
                    P_drop[t1, t2] = P_clean[t1, t2]
            Z_blk = _make_dropout_Z_tile(T_M, T_O, i0, j0)
            if Z_blk is not None:
                for t1 in range(T_M):
                    for t2 in range(T_O):
                        P_drop[t1, t2] *= Z_blk[t1, t2]; flops += 1
            # P_drop buffer now holds P_drop_ij

            # (C) Compute dV and dP_drop (dO @ V^T)
            # This is also tiled over D_Head by T_K chunks.
            dP_blk = prof.alloc_onchip((T_M, T_O), np.float32) # Accumulator for dP_drop
            for k0 in range(0, D_Head, T_K):
                dO_blk = prof.alloc_onchip((T_M, T_K), np.float32)
                V_blk  = prof.alloc_onchip((T_O, T_K), np.float32)
                for t1 in range(T_M):
                    for k in range(T_K):
                        dO_blk[t1, k] = datty_head[i0 + t1, k0 + k]
                prof.load_from_offchip(dO_blk.nbytes)
                for t2 in range(T_O):
                    for k in range(T_K):
                        V_blk[t2, k] = value_head[j0 + t2, k0 + k]
                prof.load_from_offchip(V_blk.nbytes)

                # (C-1) dV (RMW) : dV_chunk = P_drop^T @ dO_chunk
                for t2 in range(T_O):
                    for k in range(T_K):
                        acc_v = 0.0
                        for t1 in range(T_M):
                            acc_v += P_drop[t1, t2] * dO_blk[t1, k]; flops += 2
                        prof.load_from_offchip(4) # RMW load
                        d_value_head_proposed[j0 + t2, k0 + k] += acc_v; flops += 1
                        prof.store_to_offchip(4) # RMW store

                # (C-2) dP_drop accumulation: dP_blk += dO_blk @ V_blk^T
                for t1 in range(T_M):
                    for t2 in range(T_O):
                        acc = 0.0
                        for k in range(T_K):
                            acc += dO_blk[t1, k] * V_blk[t2, k]; flops += 2
                        dP_blk[t1, t2] += acc; flops += 1

                prof.free_onchip(V_blk); prof.free_onchip(dO_blk)

            # (D) dP = dP_drop ⊙ Z (This is dP_clean = dP_drop ⊙ Z)
            if Z_blk is not None:
                for t1 in range(T_M):
                    for t2 in range(T_O):
                        dP_blk[t1, t2] *= Z_blk[t1, t2]; flops += 1
                prof.free_onchip(Z_blk)
            # dP_blk buffer now holds dP_ij (or dP_clean)

            # (E) dS = P_clean ⊙ (dP − D)
            # (dP_blk is dP_ij, on_chip_D is D_i)
            dS_blk = prof.alloc_onchip((T_M, T_O), np.float32)
            for t1 in range(T_M):
                Di = on_chip_D[t1]
                for t2 in range(T_O):
                    dS_blk[t1, t2] = P_clean[t1, t2] * (dP_blk[t1, t2] - Di); flops += 2 # sub+mul

            # Free buffers that are no longer needed
            prof.free_onchip(P_clean); prof.free_onchip(P_drop); prof.free_onchip(dP_blk)

            # (F) dQ = τ·(dS @ K), (G) dK = τ·(dS^T @ Q)
            # Tiled over D_Head by T_N chunks
            for q_h_j in range(D_Head // T_N):
                col0 = q_h_j * T_N

                # Load K_sub
                K_sub = prof.alloc_onchip((T_O, T_N), np.float32)
                for t2 in range(T_O):
                    for dd in range(T_N):
                        K_sub[t2, dd] = key_head[j0 + t2, col0 + dd]
                prof.load_from_offchip(K_sub.nbytes)

                # (F) dQ (RMW) : dQ_chunk = tau * (dS_blk @ K_sub)
                for t1 in range(T_M):
                    for dd in range(T_N):
                        acc = 0.0
                        for k in range(T_O):
                            acc += dS_blk[t1, k] * K_sub[k, dd]; flops += 2  # mul+add
                        acc *= tau; flops += 1  # scale by tau
                        prof.load_from_offchip(4) # RMW load
                        d_query_head_proposed[i0 + t1, col0 + dd] += acc; flops += 1
                        prof.store_to_offchip(4) # RMW store
                prof.free_onchip(K_sub)

                # Load Q_sub
                Q_sub = prof.alloc_onchip((T_M, T_N), np.float32)
                for t1 in range(T_M):
                    for dd in range(T_N):
                        Q_sub[t1, dd] = query_head[i0 + t1, col0 + dd]
                prof.load_from_offchip(Q_sub.nbytes)

                # (G) dK (RMW) : dK_chunk = tau * (dS_blk^T @ Q_sub)
                for t2 in range(T_O):
                    for dd in range(T_N):
                        acck = 0.0
                        for t1 in range(T_M):
                            acck += dS_blk[t1, t2] * Q_sub[t1, dd]; flops += 2
                        acck *= tau; flops += 1  # scale by tau
                        prof.load_from_offchip(4) # RMW load
                        d_key_head_proposed[j0 + t2, col0 + dd] += acck; flops += 1
                        prof.store_to_offchip(4) # RMW store
                prof.free_onchip(Q_sub)

            prof.free_onchip(dS_blk)

        # Free the D_i buffer at the end of the row-block
        prof.free_onchip(on_chip_D)

    metrics = {
        "offchip_load":   prof.offchip_load_bytes,
        "offchip_store":  prof.offchip_store_bytes,
        "offchip_load_h": human_bytes(prof.offchip_load_bytes, 'MB'),
        "offchip_store_h": human_bytes(prof.offchip_store_bytes, 'MB'),
        "onchip_peak_bytes": prof.onchip_peak,
        "onchip_peak_h":     human_bytes(prof.onchip_peak, 'KB'),
        "flops": flops,
    }
    return d_query_head_proposed, d_key_head_proposed, d_value_head_proposed, metrics

# -----------------------------------------------------------
# Method B FWD: Flash Fused (Profiled) - Memory Optimization
# -----------------------------------------------------------
def compute_fwd_flash():
    """
    Flash FWD:
      - For each i-tile: keep Q_i on-chip, online-softmax row-wise to update (m,l),
        build P (clean) on the fly (not stored), apply dropout to E/P only,
        and update O = (P_drop @ V) in channel subtiles.
      - Stores O (N×D), m (N), l (N) to Off-chip.
    Profiling:
      - Off-chip I/O: call prof.load_from_offchip / store_to_offchip with exact nbytes.
      - Peak on-chip: all scratch via prof.alloc_onchip/free_onchip.
      - FLOPs: mul(+1), add(+1); exp/div excluded.
    """
    prof = Prof("flash_fwd")
    flops = 0

    # Off-chip outputs
    atty_all = np.zeros((N, D_Head), dtype=np.float32)
    m_all    = np.full((N,), -np.inf, dtype=np.float32)
    l_all    = np.zeros((N,), dtype=np.float32)

    has_mask     = ("att_mask" in globals()) and (att_mask is not None)
    p_drop       = float(globals().get("p_drop", 0.0))
    dropout_seed = int(globals().get("dropout_seed", 12345))
    tau          = float(globals().get("tau", 1.0))

    def make_dropout_tile(T_M, T_O, i0, j0):
        if p_drop <= 0.0:
            return None
        rs = np.random.RandomState((dropout_seed ^ (i0 * 1000003) ^ (j0 * 2654435761)) & 0x7FFFFFFF)
        Z = prof.alloc_onchip((T_M, T_O), np.float32)
        keep, scale = 1.0 - p_drop, 1.0 / (1.0 - p_drop)
        for t1 in range(T_M):
            for t2 in range(T_O):
                Z[t1, t2] = scale if (rs.rand() >= p_drop) else 0.0
        return Z

    print("--- Running Profiled Forward Pass (Flash, Fused) ---")

    for i0 in range(0, N, T_M):

        # Q_i on-chip
        Q_blk = prof.alloc_onchip((T_M, D_Head), np.float32)
        for t1 in range(T_M):
            Q_blk[t1, :D_Head] = query_head[i0 + t1, :D_Head]
        prof.load_from_offchip(Q_blk.nbytes)

        m_tile = prof.alloc_onchip((T_M,), np.float32); m_tile.fill(-np.inf)
        l_tile = prof.alloc_onchip((T_M,), np.float32)          # zeros
        O_tile = prof.alloc_onchip((T_M, D_Head), np.float32)   # zeros

        for j0 in range(0, N, T_O):

            # K_j on-chip
            K_full = prof.alloc_onchip((T_O, D_Head), np.float32)
            for t2 in range(T_O):
                K_full[t2, :D_Head] = key_head[j0 + t2, :D_Head]
            prof.load_from_offchip(K_full.nbytes)

            # S = τ * (Q K^T)
            S_blk = prof.alloc_onchip((T_M, T_O), np.float32)
            for t1 in range(T_M):
                for t2 in range(T_O):
                    s = 0.0
                    for d in range(D_Head):
                        s += Q_blk[t1, d] * K_full[t2, d]; flops += 2  # mul+add
                    S_blk[t1, t2] = tau * s; flops += 1                # mul (assign)

            # mask
            if has_mask:
                M_blk = prof.alloc_onchip((T_M, T_O), np.float32)
                for t1 in range(T_M):
                    for t2 in range(T_O): M_blk[t1, t2] = att_mask[i0 + t1, j0 + t2]
                prof.load_from_offchip(M_blk.nbytes)
                S_blk[M_blk <= 0.0] = -np.inf
                prof.free_onchip(M_blk)

            # online softmax (row-wise)
            rowmax = prof.alloc_onchip((T_M,), np.float32)
            for t1 in range(T_M):
                mx = m_tile[t1]
                # 💡 FIX: Changed TO to T_O
                for t2 in range(T_O):
                    if S_blk[t1, t2] > mx: mx = S_blk[t1, t2]
                rowmax[t1] = 0.0 if not np.isfinite(mx) else mx

            E_blk  = prof.alloc_onchip((T_M, T_O), np.float32)
            sumexp = prof.alloc_onchip((T_M,),     np.float32)
            for t1 in range(T_M):
                sh, se = rowmax[t1], 0.0
                for t2 in range(T_O):
                    e = np.exp(S_blk[t1, t2] - sh)  # exp not in FLOPs
                    E_blk[t1, t2] = e
                    se += e;                         flops += 1
                sumexp[t1] = se

            l_hat  = prof.alloc_onchip((T_M,), np.float32)
            l_new  = prof.alloc_onchip((T_M,), np.float32)
            invnew = prof.alloc_onchip((T_M,), np.float32)
            for t1 in range(T_M):
                r = np.exp(m_tile[t1] - rowmax[t1]) if np.isfinite(m_tile[t1]) else 0.0; flops += 2
                l_hat[t1] = l_tile[t1] * r;      flops += 1
                l_new[t1] = l_hat[t1] + sumexp[t1]; flops += 1
                invnew[t1] = 1.0 / (l_new[t1] if l_new[t1] > 0.0 else 1.0); flops += 1

            # O update with dropout: O = (O*l_hat + (E⊙Z) @ V) / l_new
            Z_blk = make_dropout_tile(T_M, T_O, i0, j0)
            for k0 in range(0, D_Head, T_K):

                V_blk = prof.alloc_onchip((T_O, T_K), np.float32)
                for t2 in range(T_O):
                    for k in range(T_K): V_blk[t2, k] = value_head[j0 + t2, k0 + k]
                prof.load_from_offchip(V_blk.nbytes)

                for t1 in range(T_M):
                    alpha = l_hat[t1] * invnew[t1];  flops += 1
                    # scale O slice by alpha
                    for k in range(T_K):
                        O_tile[t1, k0 + k] *= alpha;  flops += 1

                    for k in range(T_K):
                        acc = 0.0
                        for t2 in range(T_O):
                            g = E_blk[t1, t2]
                            if Z_blk is not None: g *= Z_blk[t1, t2]
                            acc += g * V_blk[t2, k];   flops += 2
                        
                        # 💡 FIX: (A += B*C) is 2 FLOPs (mul+add)
                        O_tile[t1, k0 + k] += invnew[t1] * acc;  flops += 2

                prof.free_onchip(V_blk)

            if Z_blk is not None: prof.free_onchip(Z_blk)

            # finalize (m,l) for this j-tile
            for t1 in range(T_M):
                m_tile[t1] = rowmax[t1]
                l_tile[t1] = l_new[t1]

            prof.free_onchip(invnew); prof.free_onchip(l_new); prof.free_onchip(l_hat)
            prof.free_onchip(sumexp); prof.free_onchip(E_blk)
            prof.free_onchip(rowmax); prof.free_onchip(S_blk); prof.free_onchip(K_full)

        # store O, m, l once per i-tile
        prof.store_to_offchip(O_tile.nbytes);  atty_all[i0:i0 + T_M, :D_Head] = O_tile[:T_M, :D_Head]
        prof.store_to_offchip(m_tile.nbytes);  m_all[i0:i0 + T_M] = m_tile[:T_M]
        prof.store_to_offchip(l_tile.nbytes);  l_all[i0:i0 + T_M] = l_tile[:T_M]

        prof.free_onchip(O_tile); prof.free_onchip(l_tile)
        prof.free_onchip(m_tile); prof.free_onchip(Q_blk)

    metrics = {
        "offchip_load":   prof.offchip_load_bytes,
        "offchip_store":  prof.offchip_store_bytes,
        "offchip_load_h": human_bytes(prof.offchip_load_bytes, 'MB'),
        "offchip_store_h":human_bytes(prof.offchip_store_bytes, 'MB'),
        "onchip_peak_bytes": prof.onchip_peak,
        "onchip_peak_h":     human_bytes(prof.onchip_peak, 'KB'),
        "flops": flops,
    }
    return atty_all, m_all, l_all, metrics

# -----------------------------------------------------------
# Method B BWD: FlashAttention style Softmax BP - CORRECTED
# -----------------------------------------------------------
def compute_bwd_flash(atty, m_all, l_all):
    prof = Prof("flash_bwd")
    flops = 0

    # Off-chip output buffers
    d_query_head_flash = np.zeros((N, D_Head), dtype=np.float32)
    d_value_head_flash = np.zeros((N, D_Head), dtype=np.float32)
    d_key_head_flash   = np.zeros((N, D_Head), dtype=np.float32)

    # Load scalar options
    has_mask     = ("att_mask" in globals()) and (att_mask is not None)
    p_drop       = float(globals().get("p_drop", 0.0))
    dropout_seed = int(globals().get("dropout_seed", 12345))
    tau          = float(globals().get("tau", 1.0))

    def _make_dropout_Z(i0, j0, TM, TO):
        if p_drop <= 0.0:
            return None
        rs = np.random.RandomState((dropout_seed ^ (i0 * 1000003) ^ (j0 * 2654435761)) & 0x7FFFFFFF)
        Z = prof.alloc_onchip((TM, TO), np.float32)
        keep = 1.0 - p_drop
        scale = 1.0 / keep
        for t1 in range(TM):
            for t2 in range(TO):
                Z[t1, t2] = scale if (rs.rand() >= p_drop) else 0.0
        return Z
    
    print("--- Running Profiled Backward Pass (Flash, Fused) ---")

    # Pre-calculate D_i = rowsum(dO_i ⊙ O_i) and store it in an Off-chip buffer (D_all)
    # This is part of the BWD pass, separated for clarity.
    D_all = np.zeros((N,), dtype=np.float32)
    for i0 in range(0, N, T_M):
        TM = min(T_M, N - i0)
        D_row = prof.alloc_onchip((TM,), np.float32)  # zeros
        for k0 in range(0, D_Head, T_K):
            TK = min(T_K, D_Head - k0)
            dO_blk = prof.alloc_onchip((TM, TK), np.float32)
            O_blk  = prof.alloc_onchip((TM, TK), np.float32)
            for t1 in range(TM):
                dO_blk[t1, :TK] = datty_head[i0 + t1, k0:k0 + TK]
                O_blk [t1, :TK] = atty      [i0 + t1, k0:k0 + TK]
            prof.load_from_offchip(dO_blk.nbytes); prof.load_from_offchip(O_blk.nbytes)
            
            for t1 in range(TM):
                acc = 0.0
                for k in range(TK):
                    acc += dO_blk[t1, k] * O_blk[t1, k];  flops += 2
                D_row[t1] += acc;  flops += 1
            prof.free_onchip(O_blk); prof.free_onchip(dO_blk)
        
        D_all[i0:i0+TM] = D_row[:TM] # Store D_i to Off-chip (D_all)
        prof.free_onchip(D_row)
    # (Off-chip store for D_all is implicitly handled by using the D_all numpy array)


    # ---------- Main BWD Pass (j-outer loop) ----------
    for j0 in range(0, N, T_O):
        TO = min(T_O, N - j0)

        # K_j on-chip (kept across i-tiles)
        K_blk_full = prof.alloc_onchip((TO, D_Head), np.float32)
        for t2 in range(TO):
            K_blk_full[t2, :D_Head] = key_head[j0 + t2, :D_Head]
        prof.load_from_offchip(K_blk_full.nbytes)

        # on-chip accumulators for this j-tile
        dK_blk = prof.alloc_onchip((TO, D_Head), np.float32)
        dV_blk = prof.alloc_onchip((TO, D_Head), np.float32)

        # iterate over i-tiles
        for i0 in range(0, N, T_M):
            TM = min(T_M, N - i0)

            # Q_i, O_i, m_i, l_i on-chip (from Off-chip)
            Q_blk = prof.alloc_onchip((TM, D_Head), np.float32)
            for t1 in range(TM):
                Q_blk[t1, :D_Head] = query_head[i0 + t1, :D_Head]
            prof.load_from_offchip(Q_blk.nbytes)

            O_blk = prof.alloc_onchip((TM, D_Head), np.float32)
            for t1 in range(TM):
                O_blk[t1, :D_Head] = atty[i0 + t1, :D_Head]
            prof.load_from_offchip(O_blk.nbytes)

            m_tile = prof.alloc_onchip((TM,), np.float32)
            l_tile = prof.alloc_onchip((TM,), np.float32)
            for t1 in range(TM):
                m_tile[t1] = m_all[i0 + t1]
                l_tile[t1] = l_all[i0 + t1]
            prof.load_from_offchip(m_tile.nbytes) 
            prof.load_from_offchip(l_tile.nbytes)
            
            # Load D_i for this row tile
            D_row = prof.alloc_onchip((TM,), np.float32)
            D_row[:TM] = D_all[i0:i0+TM]
            prof.load_from_offchip(D_row.nbytes)

            # Recompute S = τ·QK^T on (TM×TO) and apply mask
            S_blk = prof.alloc_onchip((TM, TO), np.float32)
            for t1 in range(TM):
                for t2 in range(TO):
                    s = 0.0
                    for d in range(D_Head):
                        s += Q_blk[t1, d] * K_blk_full[t2, d];  flops += 2
                    S_blk[t1, t2] = tau * s;  flops += 1
            if has_mask:
                M_blk = prof.alloc_onchip((TM, TO), np.float32)
                for t1 in range(TM):
                    M_blk[t1, :TO] = att_mask[i0 + t1, j0:j0 + TO]
                prof.load_from_offchip(M_blk.nbytes)
                S_blk[M_blk <= 0.0] = -np.inf
                prof.free_onchip(M_blk)

            # Rebuild P clean on tile: P = exp(S - m)/l (no store)
            A_blk = prof.alloc_onchip((TM, TO), np.float32)
            for t1 in range(TM):
                mrow, lrow = m_tile[t1], l_tile[t1]
                invl = 1.0 / (lrow if lrow > 0.0 else 1.0)
                for t2 in range(TO):
                    A_blk[t1, t2] = np.exp(S_blk[t1, t2] - mrow) * invl

            # Dropout mask for this (i0,j0) tile (same seeding as FWD)
            Z_blk = _make_dropout_Z(i0, j0, TM, TO)
            A_drop = None
            if Z_blk is not None:
                A_drop = prof.alloc_onchip((TM, TO), np.float32)
                for t1 in range(TM):
                    for t2 in range(TO):
                        A_drop[t1, t2] = A_blk[t1, t2] * Z_blk[t1, t2];  flops += 1

            # dP = dO @ V^T, dV = (P_drop)^T @ dO
            dP_blk = prof.alloc_onchip((TM, TO), np.float32)  # zeros
            for k0 in range(0, D_Head, T_K):
                TK = min(T_K, D_Head - k0)
                dO_blk = prof.alloc_onchip((TM, TK), np.float32)
                V_blk  = prof.alloc_onchip((TO, TK), np.float32)
                for t1 in range(TM):
                    dO_blk[t1, :TK] = datty_head[i0 + t1, k0:k0 + TK]
                prof.load_from_offchip(dO_blk.nbytes)
                for t2 in range(TO):
                    V_blk[t2, :TK] = value_head[j0 + t2, k0:k0 + TK]
                prof.load_from_offchip(V_blk.nbytes)

                # dV tile accumulation
                for t2 in range(TO):
                    for k in range(TK):
                        accv = 0.0
                        for t1 in range(TM):
                            p_used = A_blk[t1, t2] if (A_drop is None) else A_drop[t1, t2]
                            accv  += p_used * dO_blk[t1, k];  flops += 2
                        dV_blk[t2, k0 + k] += accv;  flops += 1

                # dP accumulation
                for t1 in range(TM):
                    for t2 in range(TO):
                        accP = 0.0
                        for k in range(TK):
                            accP += dO_blk[t1, k] * V_blk[t2, k];  flops += 2
                        dP_blk[t1, t2] += accP;  flops += 1

                prof.free_onchip(V_blk); prof.free_onchip(dO_blk)

            # dP_drop = dP ⊙ Z
            if Z_blk is not None:
                for t1 in range(TM):
                    for t2 in range(TO):
                        dP_blk[t1, t2] *= Z_blk[t1, t2]
                prof.free_onchip(Z_blk); prof.free_onchip(A_drop)

            # dS = P_clean ⊙ (dP_drop − D)
            dS_blk = prof.alloc_onchip((TM, TO), np.float32)
            for t1 in range(TM):
                Di = D_row[t1]
                for t2 in range(TO):
                    dS_blk[t1, t2] = A_blk[t1, t2] * (dP_blk[t1, t2] - Di);  flops += 2

            prof.free_onchip(A_blk);  prof.free_onchip(D_row); prof.free_onchip(dP_blk)
            prof.free_onchip(S_blk)

            # dQ = τ·(dS @ K_sub),  dK = τ·(dSᵀ @ Q_sub)
            for k0 in range(0, D_Head, T_N):
                TN = min(T_N, D_Head - k0)

                # K_sub from on-chip K_full (no Off-chip read)
                K_sub = prof.alloc_onchip((TO, TN), np.float32)
                for t2 in range(TO):
                    K_sub[t2, :TN] = K_blk_full[t2, k0:k0 + TN]

                # dQ RMW
                for t1 in range(TM):
                    for dd in range(TN):
                        acc = 0.0
                        for k in range(TO):
                            acc += dS_blk[t1, k] * K_sub[k, dd];  flops += 2
                        acc *= tau;  flops += 1
                        prof.load_from_offchip(4)
                        d_query_head_flash[i0 + t1, k0 + dd] += acc;  flops += 1
                        prof.store_to_offchip(4)
                prof.free_onchip(K_sub)

                # Q_sub (from on-chip Q_blk)
                Q_sub = prof.alloc_onchip((TM, TN), np.float32)
                for t1 in range(TM):
                    Q_sub[t1, :TN] = Q_blk[t1, k0:k0 + TN]
                
                # dK accumulation (on-chip)
                for t2 in range(TO):
                    for dd in range(TN):
                        acck = 0.0
                        for t1 in range(TM):
                            acck += dS_blk[t1, t2] * Q_sub[t1, dd];  flops += 2
                        acck *= tau; flops += 1
                        dK_blk[t2, k0 + dd] += acck;  flops += 1
                prof.free_onchip(Q_sub)

            prof.free_onchip(dS_blk)
            prof.free_onchip(l_tile); prof.free_onchip(m_tile)
            prof.free_onchip(O_blk);  prof.free_onchip(Q_blk)

        # store dK/dV once per j-tile
        prof.store_to_offchip(dK_blk.nbytes);  d_key_head_flash[j0:j0 + TO, :D_Head] += dK_blk[:TO, :D_Head]
        prof.store_to_offchip(dV_blk.nbytes);  d_value_head_flash[j0:j0 + TO, :D_Head] += dV_blk[:TO, :D_Head]
        prof.free_onchip(dV_blk); prof.free_onchip(dK_blk); prof.free_onchip(K_blk_full)

    metrics = {
        "offchip_load":   prof.offchip_load_bytes,
        "offchip_store":  prof.offchip_store_bytes,
        "offchip_load_h": human_bytes(prof.offchip_load_bytes, 'MB'),
        "offchip_store_h": human_bytes(prof.offchip_store_bytes, 'MB'),
        "onchip_peak_bytes": prof.onchip_peak,
        "onchip_peak_h":     human_bytes(prof.onchip_peak, 'KB'),
        "flops": flops,
    }
    return d_query_head_flash, d_key_head_flash, d_value_head_flash, metrics

# -----------------------------------------------------------
# Helpers: masked row-softmax & tile-synchronous dropout mask (matches T_M/T_O tiling)
# -----------------------------------------------------------

def _softmax_rows_masked(S: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes row-wise softmax(S). Where mask==0, the probability becomes 0 by setting S=-inf.
    Returns (P, m, l) as float32 ndarrays.
    """
    if mask is not None:
        S = S.copy()
        S[mask <= 0.0] = -np.inf
    # row max (can be -inf if a row is fully masked)
    m = np.max(S, axis=1, keepdims=True)
    m[~np.isfinite(m)] = 0.0  # guard: all-masked rows
    E = np.exp(S - m, dtype=np.float32)     # exp(-inf) -> 0
    l = E.sum(axis=1, keepdims=True)
    l = np.where(l == 0.0, 1.0, l)          # guard divide-by-zero
    P = (E / l).astype(np.float32)
    return P, m.astype(np.float32), l.astype(np.float32)

def _tile_dropout_mask(N: int, p_drop: float, seed: int) -> np.ndarray:
    """
    Generates the exact same dropout mask Z used by the tiled kernels
    by replicating their tile-based random seed generation.
    Z has values in {0, 1/(1-p)}.
    """
    if p_drop <= 0.0:
        return np.ones((N, N), dtype=np.float32)
    Z = np.empty((N, N), dtype=np.float32)
    keep_scale = 1.0 / (1.0 - p_drop)
    for i0 in range(0, N, T_M):
        for j0 in range(0, N, T_O):
            rs_seed = (seed ^ (i0 * 1000003) ^ (j0 * 2654435761)) & 0x7FFFFFFF
            rs = np.random.RandomState(rs_seed)
            tile = (rs.rand(T_M, T_O) >= p_drop).astype(np.float32) * keep_scale
            Z[i0:i0+T_M, j0:j0+T_O] = tile
    return Z

# -----------------------------------------------------------
# Baseline result Method A
# -----------------------------------------------------------
def compute_d_query_head_ref_A(P_clean, O_drop):
    """dQ_ref (Standard): Uses P_clean and O_drop passed as arguments."""
    p_drop       = float(globals().get("p_drop", 0.0))
    dropout_seed = int(globals().get("dropout_seed", 12345))

    # Load P(att) from argument
    P = P_clean.astype(np.float32)

    Z = _tile_dropout_mask(N, p_drop, dropout_seed)
    P_drop = (P * Z).astype(np.float32)

    # Load O(atty) from argument
    O = O_drop.astype(np.float32)

    # dP and dP_drop
    dP = datty_head @ value_head.T
    dP_drop = (dP * Z).astype(np.float32)

    # D (per-row scalar)
    D = (datty_head * O).sum(axis=1).astype(np.float32)

    # dS = P ⊙ (dP_drop − D[:,None])
    dS = P * (dP_drop - D[:, None])

    # dQ = dS @ K
    dQ = dS @ key_head
    return dQ.astype(np.float32)

def compute_d_key_head_ref_A(P_clean, O_drop):
    """dK_ref (Standard): Uses P_clean and O_drop passed as arguments."""
    p_drop       = float(globals().get("p_drop", 0.0))
    dropout_seed = int(globals().get("dropout_seed", 12345))

    # Load P(att) from argument
    P = P_clean.astype(np.float32)
    
    Z = _tile_dropout_mask(N, p_drop, dropout_seed)
    P_drop = (P * Z).astype(np.float32)

    # Load O(atty) from argument
    O = O_drop.astype(np.float32)
    
    dP = datty_head @ value_head.T
    dP_drop = (dP * Z).astype(np.float32)
    D = (datty_head * O).sum(axis=1).astype(np.float32)

    dS = P * (dP_drop - D[:, None])

    dK = dS.T @ query_head
    return dK.astype(np.float32)

def compute_d_value_head_ref_A(P_clean):
    """dV_ref (Standard): Uses P_clean passed as argument."""
    p_drop       = float(globals().get("p_drop", 0.0))
    dropout_seed = int(globals().get("dropout_seed", 12345))

    # Load P(att) from argument
    P = P_clean.astype(np.float32)

    Z = _tile_dropout_mask(N, p_drop, dropout_seed)
    P_drop = (P * Z).astype(np.float32)

    dV = (P_drop.T @ datty_head).astype(np.float32)
    return dV

# -----------------------------------------------------------
# Baseline result Method B
# -----------------------------------------------------------
def compute_d_query_head_ref_B():
    """dQ_ref (Ground Truth): Recomputes S, P, and O from scratch."""
    tau          = float(globals().get("tau", 1.0))
    mask         = att_mask if ("att_mask" in globals() and att_mask is not None) else None
    p_drop       = float(globals().get("p_drop", 0.0))
    dropout_seed = int(globals().get("dropout_seed", 12345))

    # Recompute P and dropout
    S = (tau * (query_head @ key_head.T)).astype(np.float32)
    P, _, _ = _softmax_rows_masked(S, mask)
    Z = _tile_dropout_mask(N, p_drop, dropout_seed)
    P_drop = (P * Z).astype(np.float32)

    # Recompute O
    O = P_drop @ value_head

    # dP and dP_drop
    dP = datty_head @ value_head.T
    dP_drop = (dP * Z).astype(np.float32)

    # D (per-row scalar)
    D = (datty_head * O).sum(axis=1).astype(np.float32)

    # dS = P ⊙ (dP_drop − D[:,None])
    dS = P * (dP_drop - D[:, None])

    # dQ = dS @ K
    # 💡 FIX: Apply tau scaling
    dQ = (dS @ key_head) * tau
    return dQ.astype(np.float32)

def compute_d_key_head_ref_B():
    """dK_ref (Ground Truth): Recomputes S, P, and O from scratch."""
    tau          = float(globals().get("tau", 1.0))
    mask         = att_mask if ("att_mask" in globals() and att_mask is not None) else None
    p_drop       = float(globals().get("p_drop", 0.0))
    dropout_seed = int(globals().get("dropout_seed", 12345))

    # Recompute P and dropout
    S = (tau * (query_head @ key_head.T)).astype(np.float32)
    P, _, _ = _softmax_rows_masked(S, mask)
    Z = _tile_dropout_mask(N, p_drop, dropout_seed)
    P_drop = (P * Z).astype(np.float32)

    # Recompute O
    O = P_drop @ value_head
    
    dP = datty_head @ value_head.T
    dP_drop = (dP * Z).astype(np.float32)
    D = (datty_head * O).sum(axis=1).astype(np.float32)

    dS = P * (dP_drop - D[:, None])

    # 💡 FIX: Apply tau scaling
    dK = (dS.T @ query_head) * tau
    return dK.astype(np.float32)

def compute_d_value_head_ref_B():
    """dV_ref (Ground Truth): Recomputes P from scratch."""
    tau          = float(globals().get("tau", 1.0))
    mask         = att_mask if ("att_mask" in globals() and att_mask is not None) else None
    p_drop       = float(globals().get("p_drop", 0.0))
    dropout_seed = int(globals().get("dropout_seed", 12345))

    # Recompute P and dropout
    S = (tau * (query_head @ key_head.T)).astype(np.float32)
    P, _, _ = _softmax_rows_masked(S, mask)
    Z = _tile_dropout_mask(N, p_drop, dropout_seed)
    P_drop = (P * Z).astype(np.float32)

    dV = (P_drop.T @ datty_head).astype(np.float32)
    return dV
    
# -----------------------------------------------------------
# Run and report (Corrected to show Individual and Total)
# -----------------------------------------------------------
if __name__ == "__main__":
    
    # --- 1. Run Pipeline A (Standard FWD + Standard BWD) ---
    print("\n=== RUNNING PIPELINE A (Standard Fused FWD + Standard BWD) ===")
    
    # 1a. Run FWD for Method A (stores atty, att)
    # (Assuming compute_fwd_proposed is an alias for compute_forward_standard_fused)
    prop_atty, prop_att, m_fwd_prop = compute_fwd_proposed() 
    
    # 1b. Run BWD for Method A (loads atty, att)
    # (Assuming compute_bwd_proposed is an alias for compute_proposed)
    d_prop_query, d_prop_key, d_prop_value, m_bwd_prop = compute_bwd_proposed(prop_atty, prop_att)
    
    
    # --- 2. Run Pipeline B (Flash FWD + Flash BWD) ---
    print("\n=== RUNNING PIPELINE B (Flash FWD + Flash BWD) ===")
    
    # 2a. Run FWD for Method B (stores atty, m, l)
    flash_atty, flash_m_all, flash_l_all, m_fwd_flash = compute_fwd_flash()
    
    # 2b. Run BWD for Method B (loads atty, m, l)
    d_flash_query, d_flash_key, d_flash_value, m_bwd_flash = compute_bwd_flash(flash_atty, flash_m_all, flash_l_all)

    
    # --- 3. Correctness Verification ---
    print("\n=== CORRECTNESS VERIFICATION ===")
    
    # Verify Method A (Standard) against ref_A
    
    # 💡 FIX: Changed prop_atty to prop_att
    d_ref_A_q = compute_d_query_head_ref_A(prop_att, prop_atty) 
    print("max query abs diff (proposed vs ref_A):", np.max(np.abs(d_prop_query - d_ref_A_q)))
    
    # 💡 FIX: Changed prop_atty to prop_att
    d_ref_A_k = compute_d_key_head_ref_A(prop_att, prop_atty) 
    print("max key abs diff (proposed vs ref_A):", np.max(np.abs(d_prop_key - d_ref_A_k)))
    
    # This line was already correct
    d_ref_A_v = compute_d_value_head_ref_A(prop_att)
    print("max value abs diff (proposed vs ref_A):", np.max(np.abs(d_prop_value - d_ref_A_v)))

    # Verify Method B (Flash) against ref_B (Ground Truth)
    d_ref_B_q = compute_d_query_head_ref_B()
    print("\nmax query abs diff (flash vs ref_B):", np.max(np.abs(d_flash_query - d_ref_B_q)))
    d_ref_B_k = compute_d_key_head_ref_B()
    print("max key abs diff (flash vs ref_B):", np.max(np.abs(d_flash_key - d_ref_B_k)))
    d_ref_B_v = compute_d_value_head_ref_B()
    print("max value abs diff (flash vs ref_B):", np.max(np.abs(d_flash_value - d_ref_B_v)))
    
    # --- 4. Metric Reporting ---
    
    # Define the helper function once
    def print_metrics(tag, m):
        print(f"\n[{tag}] metrics")
        print(f"  Off-chip loads:  {m['offchip_load_h']} MB  ({m['offchip_load']:,} B)")
        print(f"  Off-chip stores: {m['offchip_store_h']} MB ({m['offchip_store']:,} B)")
        print(f"  Peak on-chip:    {m['onchip_peak_h']} KB ({m['onchip_peak_bytes']:,} B)")
        print(f"  FLOPs:  {m['flops']:,}")

    # --- (NEW) 4a. Individual Metrics ---
    print("\n=== Proposed METRICS (FWD / BWD) ===")
    print_metrics("Proposed FWD (A)", m_fwd_prop)
    print_metrics("Proposed BWD (A)", m_bwd_prop)
    
    print("\n=== Flash METRICS (FWD / BWD) ===")
    print_metrics("Flash FWD (B)", m_fwd_flash)
    print_metrics("Flash BWD (B)", m_bwd_flash)

    # --- 4b. Final Metric Reporting (FWD + BWD) ---
    print("\n=== FINAL METRICS (FWD + BWD) ===")
    
    # Calculate totals
    total_load_prop = m_fwd_prop['offchip_load'] + m_bwd_prop['offchip_load']
    total_store_prop = m_fwd_prop['offchip_store'] + m_bwd_prop['offchip_store']
    peak_onchip_prop = max(m_fwd_prop['onchip_peak_bytes'], m_bwd_prop['onchip_peak_bytes'])
    
    m_total_prop = {
        "offchip_load": total_load_prop,
        "offchip_store": total_store_prop,
        "onchip_peak_bytes": peak_onchip_prop,
        "offchip_load_h": human_bytes(total_load_prop),
        "offchip_store_h": human_bytes(total_store_prop),
        "onchip_peak_h": human_bytes(peak_onchip_prop, 'KB'),
        "flops": m_fwd_prop['flops'] + m_bwd_prop['flops']
    }
    
    total_load_flash = m_fwd_flash['offchip_load'] + m_bwd_flash['offchip_load']
    total_store_flash = m_fwd_flash['offchip_store'] + m_bwd_flash['offchip_store']
    peak_onchip_flash = max(m_fwd_flash['onchip_peak_bytes'], m_bwd_flash['onchip_peak_bytes'])
    
    m_total_flash = {
        "offchip_load": total_load_flash,
        "offchip_store": total_store_flash,
        "onchip_peak_bytes": peak_onchip_flash,
        "offchip_load_h": human_bytes(total_load_flash),
        "offchip_store_h": human_bytes(total_store_flash),
        "onchip_peak_h": human_bytes(peak_onchip_flash, 'KB'),
        "flops": m_fwd_flash['flops'] + m_bwd_flash['flops']
    }

    print_metrics("TOTAL Proposed (A)", m_total_prop)
    print_metrics("TOTAL Flash (B)", m_total_flash)

    # --- 5. Export tables (CSV) ---------------------------------------------
    # Recompute diffs to capture in table (cheap vs whole pipeline)
    diff_prop_q = float(np.max(np.abs(d_prop_query - d_ref_A_q)))
    diff_prop_k = float(np.max(np.abs(d_prop_key   - d_ref_A_k)))
    diff_prop_v = float(np.max(np.abs(d_prop_value - d_ref_A_v)))

    diff_flash_q = float(np.max(np.abs(d_flash_query - d_ref_B_q)))
    diff_flash_k = float(np.max(np.abs(d_flash_key   - d_ref_B_k)))
    diff_flash_v = float(np.max(np.abs(d_flash_value - d_ref_B_v)))

    # 5a) rows for metrics table (FWD/BWD + TOTAL, both pipelines)
    rows_metrics = [
        {
            "pipeline": "Proposed (A)", "phase": "FWD",
            "Offchip_Load_Bytes": m_fwd_prop["offchip_load"],
            "Offchip_Load_H":     m_fwd_prop["offchip_load_h"],
            "Offchip_Store_Bytes":m_fwd_prop["offchip_store"],
            "Offchip_Store_H":    m_fwd_prop["offchip_store_h"],
            "Peak_Onchip_Bytes":  m_fwd_prop["onchip_peak_bytes"],
            "Peak_Onchip_H":      m_fwd_prop["onchip_peak_h"],
            "FLOPs": m_fwd_prop["flops"],
        },
        {
            "pipeline": "Proposed (A)", "phase": "BWD",
            "Offchip_Load_Bytes": m_bwd_prop["offchip_load"],
            "Offchip_Load_H":     m_bwd_prop["offchip_load_h"],
            "Offchip_Store_Bytes":m_bwd_prop["offchip_store"],
            "Offchip_Store_H":    m_bwd_prop["offchip_store_h"],
            "Peak_Onchip_Bytes":  m_bwd_prop["onchip_peak_bytes"],
            "Peak_Onchip_H":      m_bwd_prop["onchip_peak_h"],
            "FLOPs": m_bwd_prop["flops"],
        },
        {
            "pipeline": "Proposed (A)", "phase": "TOTAL",
            "Offchip_Load_Bytes": m_total_prop["offchip_load"],
            "Offchip_Load_H":     m_total_prop["offchip_load_h"],
            "Offchip_Store_Bytes":m_total_prop["offchip_store"],
            "Offchip_Store_H":    m_total_prop["offchip_store_h"],
            "Peak_Onchip_Bytes":  m_total_prop["onchip_peak_bytes"],
            "Peak_Onchip_H":      m_total_prop["onchip_peak_h"],
            "FLOPs": m_total_prop["flops"],
        },
        {
            "pipeline": "Flash (B)", "phase": "FWD",
            "Offchip_Load_Bytes": m_fwd_flash["offchip_load"],
            "Offchip_Load_H":     m_fwd_flash["offchip_load_h"],
            "Offchip_Store_Bytes":m_fwd_flash["offchip_store"],
            "Offchip_Store_H":    m_fwd_flash["offchip_store_h"],
            "Peak_Onchip_Bytes":  m_fwd_flash["onchip_peak_bytes"],
            "Peak_Onchip_H":      m_fwd_flash["onchip_peak_h"],
            "FLOPs": m_fwd_flash["flops"],
        },
        {
            "pipeline": "Flash (B)", "phase": "BWD",
            "Offchip_Load_Bytes": m_bwd_flash["offchip_load"],
            "Offchip_Load_H":     m_bwd_flash["offchip_load_h"],
            "Offchip_Store_Bytes":m_bwd_flash["offchip_store"],
            "Offchip_Store_H":    m_bwd_flash["offchip_store_h"],
            "Peak_Onchip_Bytes":  m_bwd_flash["onchip_peak_bytes"],
            "Peak_Onchip_H":      m_bwd_flash["onchip_peak_h"],
            "FLOPs": m_bwd_flash["flops"],
        },
        {
            "pipeline": "Flash (B)", "phase": "TOTAL",
            "Offchip_Load_Bytes": m_total_flash["offchip_load"],
            "Offchip_Load_H":     m_total_flash["offchip_load_h"],
            "Offchip_Store_Bytes":m_total_flash["offchip_store"],
            "Offchip_Store_H":    m_total_flash["offchip_store_h"],
            "Peak_Onchip_Bytes":  m_total_flash["onchip_peak_bytes"],
            "Peak_Onchip_H":      m_total_flash["onchip_peak_h"],
            "FLOPs": m_total_flash["flops"],
        },
    ]

    # 5b) rows for correctness table
    rows_diff = [
        {"pipeline":"Proposed (A)", "target":"query", "max_abs_diff": diff_prop_q},
        {"pipeline":"Proposed (A)", "target":"key",   "max_abs_diff": diff_prop_k},
        {"pipeline":"Proposed (A)", "target":"value", "max_abs_diff": diff_prop_v},
        {"pipeline":"Flash (B)",    "target":"query", "max_abs_diff": diff_flash_q},
        {"pipeline":"Flash (B)",    "target":"key",   "max_abs_diff": diff_flash_k},
        {"pipeline":"Flash (B)",    "target":"value", "max_abs_diff": diff_flash_v},
    ]
    
    # 5c) write CSVs (pandas preferred, fallback to csv module)
    import os # Import the os module to handle file paths

    # Define the output directory
    output_dir = "../Results"
    # Ensure the directory exists (create it if it doesn't)
    # os.makedirs(output_dir, exist_ok=True) 

    # Define full file paths
    metrics_file = os.path.join(output_dir, f"metrics_{N}_{D_Head}_{T_M}.csv")
    correctness_file = os.path.join(output_dir, f"correctness_{N}_{D_Head}_{T_M}.csv")

    try:
        import pandas as pd
        df_metrics = pd.DataFrame(rows_metrics)
        # Use the defined full path
        df_metrics.to_csv(metrics_file, index=False) 
        df_diff = pd.DataFrame(rows_diff)
        # Use the defined full path
        df_diff.to_csv(correctness_file, index=False)
        print(f"\nSaved tables: {metrics_file}, {correctness_file}")
    except Exception as e:
        import csv
        # metrics
        # Use the defined full path
        with open(metrics_file, "w", newline="") as f: 
            fieldnames = ["pipeline","phase","FLOPs",
                        "Offchip_Load_Bytes","Offchip_Load_H",
                        "Offchip_Store_Bytes","Offchip_Store_H",
                        "Peak_Onchip_Bytes","Peak_Onchip_H"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows_metrics)
        # diffs
        # Use the defined full path
        with open(correctness_file, "w", newline="") as f:
            fieldnames = ["pipeline","target","max_abs_diff"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows_diff)
        print(f"\nSaved tables (csv module): {metrics_file}, {correctness_file}")
