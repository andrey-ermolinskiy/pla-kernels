#pragma once

#include <cstddef>
#include <string>
#include <utility>

#include "common_defs.h"

// Sparse vector of double-precision floating point values represented as a tuple of index-value pairs.
class SparseVector final {
public:
  // Index and value of a non-zero element.
#pragma pack(push, 1)
  struct IdxValue {
    idx_t idx_;
    double value_;
  };
#pragma pack(pop)
  
  // Constructor: Initialize a sparse vector and pre-allocate space for nz_values_array.
  //   nnz_ is initialized to 0.
  //
  // Args:
  //   size:     The total number of elements in this vector (cannot exceed UINT32_MAX).
  SparseVector(size_t size);

  // Destructor
  ~SparseVector();

  // Return the dense representation of this vector as a human-readable string.
  std::string asString() const;

  // Move constructor
  SparseVector(SparseVector&&);
  
  // Copy constructor and assignment operators intentionally disabled.
  SparseVector(const SparseVector&) = delete;
  SparseVector& operator=(const SparseVector&) = delete;
  
  // The total number of elements in this vector.
  const size_t size_;

  // A heap-allocated array holding indices of non-zero elements and their values.
  // This object retains ownership of the nz_values_ array at all times.
  IdxValue* nz_values_;

  // The number of elements in the nz_values_ array.
  size_t nnz_;
};
