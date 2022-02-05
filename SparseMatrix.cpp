#include "SparseMatrix.h"

SparseMatrix::SparseMatrix(size_t rows, size_t cols, size_t nnz) :
  rows_{rows},
  cols_{cols},
  nz_values_{new double[nnz]},
  rowids_{new idx_t[nnz]},
  colptrs_{new size_t[cols + 1]},
  nnz_{nnz} { }


SparseMatrix::~SparseMatrix() {
  delete[] nz_values_;
  delete[] rowids_;
  delete[] colptrs_;
}

SparseMatrix::SparseMatrix(SparseMatrix&& src) :
  rows_{src.rows_}, cols_{src.cols_}, nz_values_{src.nz_values_}, rowids_{src.rowids_},
  colptrs_{src.colptrs_}, nnz_{src.nnz_} {
  src.nz_values_ = nullptr;
  src.rowids_ = nullptr;
  src.colptrs_ = nullptr;
}
