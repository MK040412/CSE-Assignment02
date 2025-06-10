import itertools

def held_karp(dist_matrix):
    """
    Held-Karp dynamic programming algorithm for TSP
    O(n²2ⁿ) complexity - suitable for small instances only
    """
    n = len(dist_matrix)
    
    # 실용적인 크기 제한
    if n > 15:
        raise ValueError(f"Held-Karp not suitable for {n} cities (max recommended: 15)")
    
    if n == 1:
        return [0, 0], 0
    elif n == 2:
        return [0, 1, 0], dist_matrix[0][1] + dist_matrix[1][0]
    
    # DP table: dict of {(subset_mask, endpoint): (cost, parent)}
    C = {}
    
    # Initialize: paths of length 1 from city 0
    for k in range(1, n):
        C[(1 << k, k)] = (dist_matrix[0][k], 0)
    
    # Fill DP table for increasing subset sizes
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            mask = sum(1 << s for s in subset)
            
            for j in subset:
                prev_mask = mask ^ (1 << j)  # Remove j from mask
                candidates = []
                
                for k in subset:
                    if k == j:
                        continue
                    if (prev_mask, k) in C:
                        cost = C[(prev_mask, k)][0] + dist_matrix[k][j]
                        candidates.append((cost, k))
                
                if candidates:
                    C[(mask, j)] = min(candidates)
    
    # Find optimal tour cost
    full_mask = (1 << n) - 2  # All cities except 0
    candidates = []
    
    for k in range(1, n):
        if (full_mask, k) in C:
            cost = C[(full_mask, k)][0] + dist_matrix[k][0]
            candidates.append((cost, k))
    
    if not candidates:
        raise RuntimeError("No valid tour found")
    
    opt_cost, last_city = min(candidates)
    
    # Reconstruct path
    tour = [0]
    mask = full_mask
    current = last_city
    
    while mask:
        tour.append(current)
        if (mask, current) not in C:
            break
        next_mask = mask ^ (1 << current)
        _, next_city = C[(mask, current)]
        mask = next_mask
        current = next_city
    
    tour.append(0)  # Return to start
    
    return tour, opt_cost
