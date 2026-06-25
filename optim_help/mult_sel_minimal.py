import itertools
import math


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def canonical(t):
    """
    Sorts the tuple so that (7, 3, 1) becomes (1, 3, 7).
    This ensures all permutations of the same nodes are treated as identical.
    
    Args:
        t: tuple of node indices
        
    Returns:
        Sorted tuple of the same elements
    """
    return tuple(sorted(t))


def ci_overlap(ci1, ci2):
    """
    Check if two confidence intervals overlap.
    
    Two intervals [lo1, hi1] and [lo2, hi2] overlap if there exists
    at least one point that belongs to both intervals.
    
    Args:
        ci1: tuple (lo1, hi1) - first confidence interval
        ci2: tuple (lo2, hi2) - second confidence interval
        
    Returns:
        bool: True if intervals overlap, False otherwise
        
    Example:
        [0.1, 0.5] and [0.3, 0.7] overlap → True
        [0.1, 0.3] and [0.4, 0.6] do NOT overlap → False
    """
    lo1, hi1 = ci1
    lo2, hi2 = ci2
    # Intervals overlap if the max of the lower bounds ≤ min of the upper bounds
    return max(lo1, lo2) <= min(hi1, hi2)


def extract_ci_map(data):
    """
    Parse raw data and build lookup structures for the filtering algorithm.
    
    Input data structure:
        {order: {multiplet: (omega, lo, hi, diag)}}
    where:
        - order: integer (e.g., 2 for pairs, 3 for triplets)
        - multiplet: tuple of node indices (e.g., (1, 3, 5))
        - omega: O-information value
        - lo, hi: lower and upper bounds of confidence interval
        - diag: diagnostic info
    
    Args:
        data: dict with structure described above
        
    Returns:
        multiplets_by_order: dict mapping order → sorted list of canonical multiplets
        ci_map: dict mapping canonical multiplet → (lo, hi) confidence interval
        
    Raises:
        None (silently handles missing data)
    """
    multiplets_by_order = {}
    ci_map = {}

    # Process each order level (e.g., pairs, triplets, etc.)
    for order, mp_dict in data.items():
        lst = []
        
        # Process each multiplet at this order
        for mp, (omega, lo, hi, diag) in mp_dict.items():
            # Convert multiplet to canonical form (sorted tuple)
            mp_c = canonical(mp)
            lst.append(mp_c)

            # Ensure CI bounds are floats and in correct order (lo ≤ hi)
            lo, hi = float(lo), float(hi)
            if hi < lo:
                lo, hi = hi, lo

            # Store the CI for this multiplet (keyed by canonical form)
            ci_map[mp_c] = (lo, hi)

        # Store sorted list of unique multiplets for this order
        multiplets_by_order[order] = sorted(set(lst))

    return multiplets_by_order, ci_map


# ============================================================
# CORE FILTERING ALGORITHM
# ============================================================

def select_multiplets_minimal(data):
    """
    Apply Marinazzo et al. (2024) hierarchical filtering to select multiplets.
    
    PRINCIPLE:
        A k-multiplet M is retained ONLY if its confidence interval (CI) does NOT
        overlap with the CIs of ANY of its (k-1)-subsets. If any subset's CI overlaps
        with M's CI, it indicates no additional higher-order effect, so M is discarded.
    
    ALGORITHM:
        1. Extract multiplets organized by order (pairs, triplets, etc.)
        2. For each order k, examine each k-multiplet M
        3. For smallest order: keep all (no subsets exist)
        4. For higher orders: check all (k-1)-subsets of M
        5. If ANY subset has overlapping CI → discard M
        6. If NO subsets have overlapping CI → keep M
    
    Args:
        data: dict with structure {order: {multiplet: (omega, lo, hi, diag)}}
        
    Returns:
        filtered: dict mapping order → dict of {multiplet: ""} for retained multiplets
        discarded: dict mapping multiplet → reason dict with details of why it was discarded
    """

    # Parse input and build lookup structures
    multiplets_by_order, ci_map = extract_ci_map(data)

    orders = sorted(multiplets_by_order.keys())
    
    # Initialize result dicts: will collect kept and discarded multiplets
    filtered = {k: {} for k in orders}
    discarded = {}

    # Process each order level from smallest to largest
    for k in orders:
        # Iterate through all multiplets of this order
        for M in multiplets_by_order[k]:

            # ---- STEP 1: Retrieve CI for this multiplet ----
            CI_M = ci_map.get(M)
            if CI_M is None:
                # If multiplet's CI is missing from ci_map, keep by default
                # (assumes data inconsistency should not discard multiplet)
                filtered[k][M] = ""
                continue

            # ---- STEP 2: Check if this is the smallest order ----
            if k == min(orders):
                # Smallest order has no (k-1)-subsets to compare against
                # Keep all multiplets of this order
                filtered[k][M] = ""
                continue

            # ---- STEP 3: Check all (k-1)-subsets for CI overlap ----
            overlaps = False

            # Generate all possible (k-1)-subsets of M
            # Example: if M = (1,3,5) and k=3, generate all pairs: (1,3), (1,5), (3,5)
            for S in itertools.combinations(M, k - 1):
                # Normalize subset to canonical form
                S = canonical(S)

                # Skip this subset if its CI is not in our map
                # (It may have been filtered out or not computed)
                if S not in ci_map:
                    continue

                CI_S = ci_map[S]

                # ---- STEP 4: Check for CI overlap ----
                if ci_overlap(CI_M, CI_S):
                    # If M's CI overlaps with this subset's CI, discard M
                    overlaps = True
                    discarded[M] = {
                        "order": k,
                        "CI_M": CI_M,
                        "subset": S,
                        "subset_CI": CI_S,
                        "reason": "CI-overlap"
                    }
                    break  # Stop checking subsets once we find an overlap

            # ---- STEP 5: Keep multiplet if no overlaps found ----
            if not overlaps:
                filtered[k][M] = ""

    return filtered, discarded
