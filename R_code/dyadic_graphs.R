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
dir.create(npn_mat_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dyadic_npn_dir, recursive = TRUE, showWarnings = FALSE)

# Output directories for POLY (Polychoric) method
poly_mat_dir = paste(base_dir, "GRAPHS_DIAG/POLY_mat", sep='/')      # Correlation matrices
dyadic_poly_dir = paste(base_dir, "GRAPHS_DIAG/POLY_graphs", sep='/') # Network graphs
dir.create(poly_mat_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dyadic_poly_dir, recursive = TRUE, showWarnings = FALSE)

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
    paste0(tools::file_path_sans_ext(basename(file_path)), paste0("_", name, "_polychoric.csv"))
  )
  ##############
  dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)
  ###############
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

##############
# BY THE END OF THIS PIPELINE, YOU GET THE FOLLLOWING DIR STRUCTURE:
# NPN_GRAPHS
#       DIAG_NAME_network.graphml
# NPN_MAT
#       DIAG_NAME_huge_matrix.csv 
# POLY_GRAPHS
#       DIAG_NAME_network.graphml
# POLY_MAT
#       DIAG_NAME_PSY_matrix.csv 
#
