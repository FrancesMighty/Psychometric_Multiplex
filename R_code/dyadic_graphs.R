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
# ============================================================

# ============================================================
# LOAD PACKAGES - read REQUIRED_LIBRARIES
# ============================================================

library(haven)
library(EGAnet)
library(psych)
library(Matrix)
library(igraph)
library(polycor)
library(jsonlite)
library(huge)
library(qgraph)

# ============================================================
# SETUP: DIRECTORY STRUCTURE
# ============================================================

# Base directory containing all data
base_dir = "C:/Users/utente/Documents/DataScience/TESI_MAGISTRALE/DATA"

# Input directory: raw CSV data files (one per diagnosis/condition)
data_dir = paste(base_dir, "EDI_DIAG", sep='/')

# Output directories for NPN (Nonparanormal) method
npn_mat_dir = paste(base_dir, "GRAPHS_DIAG/NPN_mat", sep='/')        # Correlation matrices
dyadic_npn_dir = paste(base_dir, "GRAPHS_DIAG/NPN_graphs", sep='/')  # Network graphs

# Output directories for POLY (Polychoric) method
poly_mat_dir = paste(base_dir, "GRAPHS_DIAG/POLY_mat", sep='/')      # Correlation matrices
dyadic_poly_dir = paste(base_dir, "GRAPHS_DIAG/POLY_graphs", sep='/') # Network graphs

# Load list of all CSV data files for each diagnosis
# Pattern: "EDI_[diagnosis].csv"
diag_data_files = list.files(data_dir, pattern = "\\.csv$", full.names = TRUE)


# ============================================================
# SECTION 1: POLYCHORIC CORRELATION & NETWORK ESTIMATION
# ============================================================
# 
# METHOD:
#   Polychoric correlation is designed for ordinal/Likert data.
#   It assumes underlying continuous latent variables.
#   Better than Pearson for discrete responses.
#
# STEPS:
#   1. Compute polychoric correlation matrix
#   2. Repair if not positive-definite 
#   3. Estimate network with EBICglasso (sparse graphical lasso with EBIC penalty)
#   4. Convert to graph format and remove zero-weight edges
#   5. Export as GraphML 
#

# Helper function: compute polychoric correlation with fallback
get_polyR = function(data, make_positive_definite = TRUE, verbose = TRUE) {
  #
  # Safely compute polychoric correlation with automatic fallback method
  #
  # ARGS:
  #   data:                     data.frame of ordinal variables
  #   make_positive_definite:   if TRUE, repair non-PD matrices
  #   verbose:                  if TRUE, print progress messages
  #
  # RETURNS:
  #   list with $R (correlation matrix) and $name (method used: 'PSY' or 'HET')
  #
  
  # --- Attempt 1: Use psych::polychoric() (preferred) ---
  if (verbose) message("Trying psych::polychoric() ...")
  result = try(psych::polychoric(data), silent = TRUE)
  
  # Check if successful
  if (inherits(result, "try-error") || is.null(result$rho)) {
    if (verbose) message("psych::polychoric() failed, retrying with polycor::hetcor() ...")
    
    # --- Fallback: Use polycor::hetcor() (heterogeneous correlation) ---
    # ML=FALSE: use two-step estimation (faster, stable)
    # use="pairwise.complete.obs": handle missing values intelligently
    result = try(polycor::hetcor(data, ML = FALSE, use = "pairwise.complete.obs"), silent = TRUE)
    if (inherits(result, "try-error")) {
      stop("Both psych::polychoric and polycor::hetcor failed.")
    }
    R = result$correlations
    name = 'HET'
  } else {
    R = result$rho
    name = 'PSY'
  }
  
  # --- Optional: Repair non-positive-definite matrices ---
  # A correlation matrix should have all eigenvalues > 0
  # Non-PD matrices can occur due to numerical estimation errors
  if (make_positive_definite) {
    eigvals = eigen(R, symmetric = TRUE, only.values = TRUE)$values
    if (any(eigvals <= 0)) {
      if (verbose) message("Matrix not positive-definite; applying nearPD() repair.")
      # nearPD(): find nearest positive-definite matrix (in Frobenius norm sense)
      R = as.matrix(Matrix::nearPD(R)$mat)
    }
  }
  
  if (verbose) message("get_polyR() completed successfully.")
  return(list(R=R, name=name))
}


# --- Process each diagnosis file ---
for (file_path in diag_data_files) {
  # Extract diagnosis name from filename
  diag = tools::file_path_sans_ext(basename(file_path))
  print(paste(file_path, diag, sep = ', '))
  
  # ---- Read and prepare data ----
  # [2:92] selects the 91 EDI items (skip ID column)
  data = read.csv(file_path)[2:92]
  
  # ---- Compute polychoric correlation ----
  result = get_polyR(data)
  poly_cor = result$R
  name = result$name
  
  # ---- Validate: Check positive-definiteness ----
  EDI_eigen_values = eigen(poly_cor)$values
  is_pos_def = all(EDI_eigen_values > 0)
  
  cat("\nFile:", basename(file_path),
      "\nPositive definite:", is_pos_def, "\n")
  
  # ---- Export: Correlation matrix ----
  out_file = file.path(
    poly_mat_dir,
    paste0(tools::file_path_sans_ext(basename(file_path)), paste0(name, "_polychoric.csv"))
  )
  write.csv(poly_cor, out_file, row.names = TRUE)
  cat('Poly corr done\n')
  
  # ---- Network estimation: EBICglasso ----
  # EBICglasso: Sparse graphical lasso with Extended BIC penalty
  # S: correlation/covariance matrix (input)
  # n: sample size
  # gamma: EBIC parameter (0.1 = moderate sparsity; higher = sparser network)
  # Returns: sparse partial correlation matrix (network adjacency)
  network = EBICglasso(
    S = poly_cor,
    n = nrow(data),
    gamma = 0.1
  )
  
  # ---- Convert to graph object ----
  # Create igraph object from adjacency matrix
  # mode="undirected": treat edges as bidirectional (no directionality)
  # weighted=TRUE: preserve edge weights (partial correlations)
  # diag=FALSE: ignore diagonal (self-loops)
  g = graph_from_adjacency_matrix(
    network,
    mode = "undirected",
    weighted = TRUE,
    diag = FALSE
  )
  
  # ---- Clean: Remove zero-weight edges ----
  # After EBICglasso, many edges are exactly 0 (sparse solution)
  # Delete these non-edges to reduce noise
  g = delete_edges(g, which(E(g)$weight == 0))
  print(ecount(g))       # Print number of edges
  print(is.connected(g)) # Check if graph is connected
  
  # ---- Export: GraphML format ----
  # GraphML: XML-based graph format readable by Gephi, Cytoscape, etc.
  net_sub_path = paste(diag, "network.graphml", sep='_' )
  write_graph(g, file.path(dyadic_poly_dir, net_sub_path), format = "graphml")
}



# ============================================================
# SECTION 2: NONPARANORMAL (NPN) / GAUSSIAN COPULA METHOD
# ============================================================
#
# METHOD:
#   Nonparanormal transformation: rank-based Gaussian copula approach
#   Converts ordinal data to approximately Gaussian while preserving
#   dependence structure. Then use standard Pearson correlation.
#
# ADVANTAGES:
#   - Theoretically sound for ordinal data
#   - Computationally efficient
#   - Doesn't require assuming distributional form
#
# STEPS:
#   1. Apply huge.npn() transformation (shrinkage-based van der Waerst)
#   2. Compute Pearson correlation on transformed data
#   3. Network estimation with EBICglasso (same as above)
#   4. Graph conversion and export
#

for (file_path in diag_data_files) {
  # Extract diagnosis name
  diag = tools::file_path_sans_ext(basename(file_path))
  print(paste(file_path, diag, sep = ', '))
  
  # ---- Read and convert to matrix ----
  data = read.csv(file_path)[2:92]
  # sapply(data, as.numeric): ensure all columns are numeric
  X_mat <- as.matrix(sapply(data, as.numeric))
    
  # ---- Apply Nonparanormal transformation ----
  # huge.npn(): Gaussian copula transformation
  # Approximately linearizes rank-based relationships
  # npn.func = "shrinkage": shrinkage-based van der Waerst transform
  #   (stable, recommended; "truncation" is also available but less stable)
  X_npn <- huge.npn(X_mat, npn.func = "shrinkage")
  
  # ---- Compute Pearson correlation on transformed data ----
  # After NPN transformation, Pearson correlation is appropriate
  R_npn <- cor(X_npn)
  
  # ---- Export: Correlation matrix ----
  npn_file_name = paste(diag, "huge_matrix.csv", sep='_' )
  write.csv(R_npn, file.path(npn_mat_dir, npn_file_name), row.names = TRUE)
  
  # ---- Network estimation: EBICglasso ----
  # Same sparse network estimation as polychoric method
  EBIC_fit <- EBICglasso(R_npn, 
                         n = nrow(X_mat), 
                         gamma = 0.1)
  
  # ---- Convert to graph object ----
  g = graph_from_adjacency_matrix(
    EBIC_fit,
    mode = "undirected",
    weighted = TRUE,
    diag = FALSE
  )
  
  # ---- Clean: Remove zero-weight edges ----
  g = delete_edges(g, which(E(g)$weight == 0))
  print(ecount(g))       # Print edge count
  print(is.connected(g)) # Check connectivity
    
  # ---- Export: GraphML format ----
  net_sub_path = paste(diag, "network.graphml", sep='_' )
  write_graph(g, file.path(dyadic_npn_dir, net_sub_path), format = "graphml")
}



# ============================================================
# SECTION 3: REDUNDANCY ANALYSIS VIA UVA
# ============================================================
#
# METHOD:
#   UVA = Unique Validity Analysis
#   Identifies "redundant" nodes (variables) that don't contribute
#   unique information to the network structure.
#
# INTERPRETATION:
#   Redundant nodes: highly replaceable by their neighbors
#                    (low unique variance explained)
#   Essential nodes:  essential to network structure
#
# NOTE:
#   Applied to both NPN and POLY correlation matrices separately
#

# Initialize lists to store redundancy results
# Keys = diagnosis names, Values = redundant node indices
red_NPN_UVA = vector("list", length(npn_mats))
names(red_NPN_UVA) = tools::file_path_sans_ext(basename(npn_mats))

red_POLY_UVA = vector("list", length(poly_mats))
names(red_POLY_UVA) = tools::file_path_sans_ext(basename(poly_mats))

# Load list of correlation matrices from disk
# (outputs from Sections 1 and 2)
npn_mats = list.files(npn_mat_dir, pattern = "\\.csv$", full.names = TRUE) 
poly_mats = list.files(poly_mat_dir, pattern = "\\.csv$", full.names = TRUE)   


# --- NPN METHOD: Identify redundant nodes ---
for (file_path in npn_mats) {
  diag = tools::file_path_sans_ext(basename(file_path))
  # Read correlation matrix (skip row names)
  R = read.csv(file_path)[2:92]
  print(diag)
  
  # UVA: Unique Validity Analysis
  # data:          correlation/covariance matrix
  # n:             sample size (1206 subjects)
  # type:          "ordinal" (appropriate for Likert data)
  # uva.method:    "MBR" = Minimum Bottleneck Rerouting network method
  # cut.off:       significance threshold for redundancy
  # reduce.method: 'latent' = latent variable reduction method
  uva_results = UVA(
    data = R,
    n = 1206,
    type = "ordinal",
    uva.method = "MBR",
    cut.off = cutoff_val,           
    reduce.method = 'latent'
  )
  
  # Extract list of redundant nodes for this diagnosis
  red_NPN_UVA[[diag]] = uva_results$redundant
}

# ---- Export: NPN redundancy results ----
write_json(red_NPN_UVA, "red_NPN_UVA.json", pretty = TRUE)


# --- POLY METHOD: Identify redundant nodes ---
for (file_path in poly_mats) {
  diag = tools::file_path_sans_ext(basename(file_path))
  R = read.csv(file_path)[2:92]
  print(diag)
  
  # Same UVA analysis on polychoric correlations
  uva_results = UVA(
    data = R,
    n = 1206,
    type = "ordinal",
    uva.method = "MBR",
    cut.off = cutoff_val,           
    reduce.method = 'latent'
  )
  
  # Extract redundant nodes
  red_POLY_UVA[[diag]] = uva_results$redundant
}

# ---- Export: POLY redundancy results ----
write_json(red_POLY_UVA, "red_POLY_UVA.json", pretty = TRUE)




# ============================================================
# SECTION 4: GENERAL NETWORK (OVERALL SAMPLE)
# ============================================================
#
# PURPOSE:
#   Create a single network graph from the full EDI dataset
#   (all 1206 subjects combined, not separated by diagnosis)
#
# NOTE:
#   Uses NPN method (same as Section 2)
#

# ---- Load full dataset ----
# Read from SPSS format (.sav)
EDI_data = read_sav("~/DataScience/TESI_MAGISTRALE/DATA/EDI-3/EDI-3-merged-overall-sample_5-pt-scoring--n-1206-.sav")
# Sort by PatientID for consistency
EDI_data = EDI_data[order(EDI_data$PatientID), ]
# Select EDI items (columns 2:92)
EDI_items = EDI_data[2:92]

# ---- Convert to numeric matrix ----
X_mat <- as.matrix(sapply(EDI_items, as.numeric))

# ---- Apply Nonparanormal transformation ----
X_npn <- huge.npn(X_mat, npn.func = "shrinkage")

# ---- Compute Pearson correlation on transformed data ----
R_npn <- cor(X_npn)

# ---- Export: Correlation matrix ----
write.csv(R_npn, "huge_matrix_ALL.csv", row.names = TRUE)

# ---- Network estimation: EBICglasso ----
EBIC_fit <- EBICglasso(R_npn, 
                       n = nrow(X_mat), 
                       gamma = 0.1)

# ---- Convert to graph object ----
g = graph_from_adjacency_matrix(
  EBIC_fit,
  mode = "undirected",
  weighted = TRUE,
  diag = FALSE
)

# ---- Clean: Remove zero-weight edges ----
g = delete_edges(g, which(E(g)$weight == 0))
print(ecount(g))       # Print edge count
print(is.connected(g)) # Check connectivity

# ---- Export: GraphML format ----
# Final output: network graph for all 1206 subjects
write_graph(g, "network_ALL.graphml", format = "graphml")

# ============================================================
# END OF PIPELINE
# ============================================================
