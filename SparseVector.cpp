#include "SparseVector.h"

#include <sstream>
#include <limits>

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


std::string SparseVector::asString() const {
   std::ostringstream ret;
   ret << "[";
   size_t cur_idx = 0;
   for (size_t i = 0; i < size_; ++i) {
     if (cur_idx < nnz_ && nz_values_[cur_idx].idx_ == i)
       ret << nz_values_[cur_idx++].value_;
     else
       ret << '0';
     if (i < (size_ - 1))
       ret << ' ';
   }
   ret << "]";  
   return ret.str();      
}
