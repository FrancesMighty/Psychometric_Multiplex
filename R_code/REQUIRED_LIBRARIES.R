# ============================================================
# REQUIRED R LIBRARIES FOR NPN_POLY_GRAPHS.R
# ============================================================

# Installation command (run once):
# install.packages(c("psych", "polycor", "Matrix", "igraph", "huge", 
#                    "glasso", "jsonlite", "haven", "qgraph"))


# ============================================================
# LIBRARY DETAILS
# ============================================================

# SECTION 1: POLYCHORIC CORRELATION
# ============================================================

# psych
#   Purpose: Compute polychoric correlation matrices
#   Function: psych::polychoric()
#   Use: Primary method for ordinal/Likert data
#   Install: install.packages("psych")

# polycor
#   Purpose: Alternative heterogeneous correlation estimation
#   Function: polycor::hetcor()
#   Use: Fallback if psych::polychoric() fails
#   Install: install.packages("polycor")

# Matrix
#   Purpose: Advanced matrix operations and repair
#   Function: Matrix::nearPD() - find nearest positive-definite matrix
#   Use: Fix non-positive-definite correlation matrices
#   Install: install.packages("Matrix")


# SECTION 2: NETWORK ESTIMATION & GRAPHICAL LASSO
# ============================================================

# EGAnet (qgraph) or glasso
#   Purpose: Sparse graphical lasso with EBIC penalty
#   Function: EBICglasso()
#   Use: Estimate sparse network (partial correlations)
#   Note: EBICglasso() is typically in 'glasso' or 'qgraph' package
#         qgraph is more comprehensive but heavier; glasso is focused
#   Install: install.packages("glasso")
#       OR: install.packages("qgraph")

# igraph
#   Purpose: Graph/network object creation and manipulation
#   Functions: 
#     - graph_from_adjacency_matrix(): convert matrix to graph
#     - delete_edges(): remove edges
#     - E(): access edge attributes
#     - ecount(): count edges
#     - is.connected(): check connectivity
#     - write_graph(): export to various formats
#   Use: Create, manipulate, and export network graphs
#   Install: install.packages("igraph")


# SECTION 3: NONPARANORMAL TRANSFORMATION
# ============================================================

# huge
#   Purpose: High-dimensional Undirected Graphical Estimation
#   Function: huge.npn() - Nonparanormal (Gaussian copula) transformation
#   Use: Transform ordinal data to approximately Gaussian while preserving dependence
#   Install: install.packages("huge")


# SECTION 4: REDUNDANCY ANALYSIS
# ============================================================

# (No additional libraries beyond those listed above for UVA function)
# Note: UVA() may be a custom function or from 'qgraph' package


# SECTION 5: DATA I/O & UTILITIES
# ============================================================

# jsonlite
#   Purpose: JSON file I/O
#   Function: write_json()
#   Use: Export redundancy analysis results as JSON
#   Install: install.packages("jsonlite")

# haven
#   Purpose: Read/write SPSS, Stata, and SAS files
#   Function: read_sav() - read SPSS .sav files
#   Use: Load the full EDI-3 dataset
#   Install: install.packages("haven")

# tools (BASE R - no installation needed)
#   Purpose: File path utilities
#   Functions:
#     - tools::file_path_sans_ext(): remove file extension
#     - basename(): get filename from full path
#   Use: Extract diagnosis names from filenames


# ============================================================
# COMPLETE INSTALLATION COMMAND
# ============================================================

install.packages(c(
  "psych",      # Polychoric correlation
  "polycor",    # Alternative polychoric method
  "Matrix",     # Matrix operations
  "igraph",     # Network graphs
  "huge",       # Nonparanormal transformation
  "EGAnet",     # EBIC graphical lasso
  "jsonlite",   # JSON export
  "haven"       # SPSS file I/O
))



# ============================================================
# LOAD LIBRARIES IN SCRIPT
# ============================================================

# At the beginning of NPN_POLY_GRAPHS.R, add:

library(psych)      # Polychoric correlation
library(polycor)    # Alternative polychoric
library(Matrix)     # Matrix operations
library(igraph)     # Network manipulation
library(huge)       # Nonparanormal
library(EGAnet)     # EBIC glasso 
# library(qgraph)   # Uncomment if using qgraph version
library(jsonlite)   # JSON I/O
library(haven)      # SPSS file reading


# ============================================================
# CHECKING INSTALLATION
# ============================================================

# Verify all packages are installed:
required_packages <- c("psych", "polycor", "Matrix", "igraph", "huge", 
                       "glasso", "jsonlite", "haven")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    cat(paste("Package", pkg, "not installed. Installing...\n"))
    install.packages(pkg)
  }
}
