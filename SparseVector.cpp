#include "SparseVector.h"

#include <sstream>

SparseVector::SparseVector(size_t size) :
  size_{size},
  nz_values_{new IdxValue[size]},
  nnz_{0} {
  if (size_ > std::numeric_limits<uint32_t>::max())
    fatal("Sparse vector size exceeds the largest supported value.");
}


SparseVector::~SparseVector() {
  delete[] nz_values_;
}


SparseVector::SparseVector(SparseVector&& src) :
  size_{src.size_},
  nz_values_{src.nz_values_},
  nnz_{src.nnz_} {
    src.nz_values_ = nullptr;
}
