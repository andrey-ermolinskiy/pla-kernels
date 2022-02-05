#pragma once

#include "SparseMatrix.h"
#include "SparseVector.h"

// Implementation of a parallel SpMSpV algorithm.
//
// Reference:
//   Ariful Azad, & Aydin Buluc. (2016). "A work-efficient parallel sparse matrix-sparse vector multiplication algorithm."

class SpMSpV {
public: // --------------------------------- Public interface -----------------------------------------------
  // Constructor: Initialize the multipler for the specified problem size and allocate the necessary resources.
  //
  // Args:
  //   rows: Number of rows in the input matrix.
  //   cols: Number of columns in the input matrix.
  SpMSpV(size_t rows, size_t cols);

  // Destructor.
  ~SpMSpV();

  // Copy constructor and assignment operators intentionally disabled.
  SpMSpV(const SpMSpV&) = delete;
  SpMSpV& operator=(const SpMSpV&) = delete;

  // Compute y = a * x and return y
  //
  // Args:
  //   a:   Input matrix (first operand).
  //        The number of non-zero elements in this matrix may not exceed MAX_MATRIX_NNZ.
  //        The matrix is assumed not to contain NaN values.  
  //   x:   Input vector (second operand).
  //   y:   Output vector. y.nz_values_ is assumed to be large enough to hold a.rows_ elements.
  //
  void multiply(const SparseMatrix& a, const SparseVector& x, SparseVector* y);

  // Upper bound on the number of non-zero elements in the input matrix.  
  static constexpr size_t MAX_MATRIX_NNZ = 1024 * 1024;

  // Number of worker threads.
  static constexpr int NUM_THREADS = 4;

  // In order to balance work among threads, we bucketize the workload and create more buckets than the
  // available number of threads.
  static constexpr int NUM_BUCKETS = 8 * NUM_THREADS;
  
private: // ------------------------ Private interface and data members -------------------------------
  // Data structure that stores per-bucket state for parallel steps 2 and 3.
  struct alignas(64) BucketState {
    // The starting offset of this bucket in the scaled_values_ array.
    size_t start_offset_;

    // The number of elements in this bucket.
    size_t size_;

    // The number of valid elements in the uinds_ array.
    size_t num_uinds_;

    // A list of unique output vector indices.
    idx_t uinds_[(MAX_MATRIX_NNZ + (NUM_BUCKETS - 1)) / NUM_BUCKETS];        
  };
  
  // A pre-allocated memory region holding a sequence of <i, A(i,j) * x(j)> pairs. This sequence is
  // divided into variable-length buckets.
  SparseVector::IdxValue* const scaled_values_;

  // An array of bucket state objects.
  BucketState* const bucket_state_array_;

  // A pre-allocated sparse accumulator.
  double* const spa_;
};
