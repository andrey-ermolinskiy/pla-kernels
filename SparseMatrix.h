#pragma once

#include "common_defs.h"
#include "SparseVector.h"

// Sparse matrix of double-precision floating point values represented in CSC (Compressed Sparse Column) form.
class SparseMatrix final {
public:
  // Constructor: Initialize a sparse matrix and allocate space for the specified number of non-zero elements.
  //   The nz_values_, rowids_, and colptrs_ arrays remain uninitialized.
  //
  // Args:
  //   rows:  The number of rows in this matrix (cannot exceed UINT32_MAX).
  //   cols:  The number of columns in this matrix (cannot exceed UINT32_MAX).
  SparseMatrix(size_t rows, size_t cols, size_t nnz);

  // Destructor.
  ~SparseMatrix();

  // Move constructor
  SparseMatrix(SparseMatrix&&);

  // Copy constructor and assignment operators intentionally disabled
  SparseMatrix(const SparseMatrix&) = delete;
  SparseMatrix& operator=(const SparseMatrix&) = delete;

  // The number of rows in this matrix.
  const size_t rows_;

  // The number of columns in this matrix.
  const size_t cols_;
  
  // A heap-allocated array of length nnz_ storing the values of non-zero elements. Non-zero elements of a column
  // occupy consecutive positions in this array. [*]
  double* nz_values_;

  // A heap-allocated array of length nnz_. [*]
  // rowids_[i] stores the row index of a non-zero element with value nz_values_[i].
  idx_t* rowids_;

  // A heap-allocated array of length (cols_ + 1). [*]
  // For any given column index c, colptrs_[c] stores the location of the first nonzero element of column c in the
  // nz_values_ array. The number of non-zero elements in column c is given by (colptrs_[c + 1] - colptrs_[c]).
  size_t* colptrs_;

  // The number of elements in the nz_values_ and rowids_ arrays.
  const size_t nnz_;

  // [*] This object retains memory ownership at all times.
};
