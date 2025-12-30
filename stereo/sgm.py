import numpy as np
from numba import njit

INF = 1 << 30

def sgm_aggregate_8dir(cost_vol: np.ndarray,
                       P1: int = 2,
                       P2: int = 24) -> np.ndarray:
    """
    cost_vol: (H, W, D)
    return S: (H, W, D), int32 aggregated cost
    8 directions: → ← ↓ ↑  ↘ ↖ ↗ ↙
    """
    assert cost_vol.ndim == 3
    H, W, D = cost_vol.shape
    C = cost_vol.astype(np.int32)
    S = np.zeros((H, W, D), dtype=np.int32)

    def dp_update(prev_L: np.ndarray, c: np.ndarray) -> np.ndarray:
        min_prev = prev_L.min()

        prev_minus = np.empty_like(prev_L)
        prev_plus  = np.empty_like(prev_L)

        prev_minus[0] = INF
        prev_minus[1:] = prev_L[:-1]

        prev_plus[-1] = INF
        prev_plus[:-1] = prev_L[1:]

        t0 = prev_L
        t1 = prev_minus + P1
        t2 = prev_plus + P1
        t3 = min_prev + P2

        m = np.minimum(t0, np.minimum(t1, np.minimum(t2, t3)))
        return c + m - min_prev

    # -------- 1) left -> right (→) --------
    for y in range(H):
        L_prev = np.zeros(D, np.int32)
        for x in range(W):
            L = C[y, x].copy() if x == 0 else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L

    # -------- 2) right -> left (←) --------
    for y in range(H):
        L_prev = np.zeros(D, np.int32)
        for x in range(W - 1, -1, -1):
            L = C[y, x].copy() if x == W - 1 else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L

    # -------- 3) top -> bottom (↓) --------
    for x in range(W):
        L_prev = np.zeros(D, np.int32)
        for y in range(H):
            L = C[y, x].copy() if y == 0 else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L

    # -------- 4) bottom -> top (↑) --------
    for x in range(W):
        L_prev = np.zeros(D, np.int32)
        for y in range(H - 1, -1, -1):
            L = C[y, x].copy() if y == H - 1 else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L

    # -------- 5) top-left -> bottom-right (↘) --------
    # scan order: y increasing, x increasing
    for y0 in range(H):
        y, x = y0, 0
        L_prev = np.zeros(D, np.int32)
        first = True
        while y < H and x < W:
            L = C[y, x].copy() if first else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L
            first = False
            y += 1; x += 1
    for x0 in range(1, W):
        y, x = 0, x0
        L_prev = np.zeros(D, np.int32)
        first = True
        while y < H and x < W:
            L = C[y, x].copy() if first else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L
            first = False
            y += 1; x += 1

    # -------- 6) bottom-right -> top-left (↖) --------
    # reverse of ↘ : y decreasing, x decreasing
    for y0 in range(H - 1, -1, -1):
        y, x = y0, W - 1
        L_prev = np.zeros(D, np.int32)
        first = True
        while y >= 0 and x >= 0:
            L = C[y, x].copy() if first else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L
            first = False
            y -= 1; x -= 1
    for x0 in range(W - 2, -1, -1):
        y, x = H - 1, x0
        L_prev = np.zeros(D, np.int32)
        first = True
        while y >= 0 and x >= 0:
            L = C[y, x].copy() if first else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L
            first = False
            y -= 1; x -= 1

    # -------- 7) bottom-left -> top-right (↗) --------
    # scan order: y decreasing, x increasing
    for y0 in range(H - 1, -1, -1):
        y, x = y0, 0
        L_prev = np.zeros(D, np.int32)
        first = True
        while y >= 0 and x < W:
            L = C[y, x].copy() if first else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L
            first = False
            y -= 1; x += 1
    for x0 in range(1, W):
        y, x = H - 1, x0
        L_prev = np.zeros(D, np.int32)
        first = True
        while y >= 0 and x < W:
            L = C[y, x].copy() if first else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L
            first = False
            y -= 1; x += 1

    # -------- 8) top-right -> bottom-left (↙) --------
    # reverse of ↗ : y increasing, x decreasing
    for y0 in range(H):
        y, x = y0, W - 1
        L_prev = np.zeros(D, np.int32)
        first = True
        while y < H and x >= 0:
            L = C[y, x].copy() if first else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L
            first = False
            y += 1; x -= 1
    for x0 in range(W - 2, -1, -1):
        y, x = 0, x0
        L_prev = np.zeros(D, np.int32)
        first = True
        while y < H and x >= 0:
            L = C[y, x].copy() if first else dp_update(L_prev, C[y, x])
            S[y, x] += L
            L_prev = L
            first = False
            y += 1; x -= 1

    return S

@njit(cache=True)
def dp_update_1d(prev_L, c, P1, P2, out_L):
    D = prev_L.shape[0]

    # min_prev
    min_prev = prev_L[0]
    for d in range(1, D):
        v = prev_L[d]
        if v < min_prev:
            min_prev = v

    base_jump = min_prev + P2

    for d in range(D):
        best = prev_L[d]

        if d > 0:
            v = prev_L[d-1] + P1
            if v < best:
                best = v

        if d < D - 1:
            v = prev_L[d+1] + P1
            if v < best:
                best = v

        if base_jump < best:
            best = base_jump

        out_L[d] = c[d] + best - min_prev


@njit(cache=True)
def sgm_aggregate_8dir_numba(cost_vol, P1, P2):
    """
    cost_vol: (H, W, D)
    return S: (H, W, D) int32
    8 dirs: → ← ↓ ↑  ↘ ↖ ↗ ↙
    """
    H, W, D = cost_vol.shape
    C = cost_vol.astype(np.int32)
    S = np.zeros((H, W, D), dtype=np.int32)

    L_prev = np.zeros(D, dtype=np.int32)
    L_cur  = np.zeros(D, dtype=np.int32)

    # ---------- 1) left -> right (→) ----------
    for y in range(H):
        for d in range(D): L_prev[d] = 0
        for x in range(W):
            if x == 0:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

    # ---------- 2) right -> left (←) ----------
    for y in range(H):
        for d in range(D): L_prev[d] = 0
        for x in range(W-1, -1, -1):
            if x == W-1:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

    # ---------- 3) top -> bottom (↓) ----------
    for x in range(W):
        for d in range(D): L_prev[d] = 0
        for y in range(H):
            if y == 0:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

    # ---------- 4) bottom -> top (↑) ----------
    for x in range(W):
        for d in range(D): L_prev[d] = 0
        for y in range(H-1, -1, -1):
            if y == H-1:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

    # ---------- 5) top-left -> bottom-right (↘) ----------
    # starts on left edge (y0,0)
    for y0 in range(H):
        for d in range(D): L_prev[d] = 0
        y = y0
        x = 0
        first = True
        while y < H and x < W:
            if first:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
                first = False
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

            y += 1
            x += 1

    # starts on top edge (0,x0), x0>=1
    for x0 in range(1, W):
        for d in range(D): L_prev[d] = 0
        y = 0
        x = x0
        first = True
        while y < H and x < W:
            if first:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
                first = False
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

            y += 1
            x += 1

    # ---------- 6) bottom-right -> top-left (↖) ----------
    # starts on right edge (y0,W-1), y0 from H-1..0
    for y0 in range(H-1, -1, -1):
        for d in range(D): L_prev[d] = 0
        y = y0
        x = W - 1
        first = True
        while y >= 0 and x >= 0:
            if first:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
                first = False
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

            y -= 1
            x -= 1

    # starts on bottom edge (H-1,x0), x0<=W-2
    for x0 in range(W-2, -1, -1):
        for d in range(D): L_prev[d] = 0
        y = H - 1
        x = x0
        first = True
        while y >= 0 and x >= 0:
            if first:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
                first = False
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

            y -= 1
            x -= 1

    # ---------- 7) bottom-left -> top-right (↗) ----------
    # starts on left edge (y0,0), y0 from H-1..0
    for y0 in range(H-1, -1, -1):
        for d in range(D): L_prev[d] = 0
        y = y0
        x = 0
        first = True
        while y >= 0 and x < W:
            if first:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
                first = False
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

            y -= 1
            x += 1

    # starts on bottom edge (H-1,x0), x0>=1
    for x0 in range(1, W):
        for d in range(D): L_prev[d] = 0
        y = H - 1
        x = x0
        first = True
        while y >= 0 and x < W:
            if first:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
                first = False
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

            y -= 1
            x += 1

    # ---------- 8) top-right -> bottom-left (↙) ----------
    # starts on right edge (y0,W-1), y0 from 0..H-1
    for y0 in range(H):
        for d in range(D): L_prev[d] = 0
        y = y0
        x = W - 1
        first = True
        while y < H and x >= 0:
            if first:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
                first = False
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

            y += 1
            x -= 1

    # starts on top edge (0,x0), x0<=W-2
    for x0 in range(W-2, -1, -1):
        for d in range(D): L_prev[d] = 0
        y = 0
        x = x0
        first = True
        while y < H and x >= 0:
            if first:
                for d in range(D):
                    L_cur[d] = C[y, x, d]
                first = False
            else:
                dp_update_1d(L_prev, C[y, x, :], P1, P2, L_cur)

            for d in range(D):
                S[y, x, d] += L_cur[d]
                L_prev[d] = L_cur[d]

            y += 1
            x -= 1

    return S


def sgm_disparity(cost_vol: np.ndarray, P1: int = 8, P2: int = 32) -> np.ndarray:
    S = sgm_aggregate_8dir(cost_vol,P1=P1, P2=P2)
    disp = np.argmin(S, axis=2).astype(np.uint16)
    return disp

def sgm_disparity_8dir_numba(cost_vol, P1=8, P2=32):
    S = sgm_aggregate_8dir_numba(cost_vol, P1, P2)
    disp = np.argmin(S, axis=2).astype(np.uint16)
    return disp

