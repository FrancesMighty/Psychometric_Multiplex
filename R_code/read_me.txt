# ============================================================
# NETWORK ANALYSIS PIPELINE FOR EDI DATA
# ============================================================
# 
# PURPOSE:
#   Construct network graphs from Eating Disorder Inventory (EDI)
#   ordinal/Likert response data using two correlation methods:
#   1. Polychoric correlation (appropriate for ordinal data)
#   2. Nonparanormal transformation (Gaussian copula method)
#
# PIPELINE:
#   Raw data → Correlation matrices → Network estimation (EBICglasso) 
#   → Redundancy analysis (UVA) → GraphML export
#
# DATA:
#   EDI-3 dataset with 1206 subjects and 91 items (columns 2:92)
#
