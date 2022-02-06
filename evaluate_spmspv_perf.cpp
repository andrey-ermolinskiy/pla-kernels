#include <vector>
#include <cmath>
#include <chrono>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unordered_set>

#include "SparseMatrix.h"
#include "SparseVector.h"
#include "SpMSpV.h"

// TODO: remove ths method
static std::vector<std::vector<double>> sparseMatrixToDense(const SparseMatrix &mat) {
  std::vector<std::vector<double>> m_dense(mat.rows_);
  for (size_t i = 0; i < mat.rows_; ++i) {
    m_dense[i].resize(mat.cols_, 0);
  }

  size_t idx = 0;
  for (size_t j = 0; j < mat.cols_; ++j) {
    size_t nnz_in_col = mat.colptrs_[j + 1] - mat.colptrs_[j];
    for (size_t t = 0; t  <nnz_in_col; ++ t) {
      double val = mat.nz_values_[idx];
      idx_t i = mat.rowids_[idx];
      m_dense[i][j] = val;
      idx++;
    }
  }

  return m_dense;
}

// TODO: remove this method
static std::vector<double> sparseVectorToDense(const SparseVector &v) {
  std::vector<double> v_dense(v.size_, 0);
  for (size_t i = 0; i < v.nnz_; ++i)
    v_dense[v.nz_values_[i].idx_] = v.nz_values_[i].value_;
  return v_dense;
}

// TODO: remove this method
static std::vector<double> denseMultiply(const std::vector<std::vector<double>> &a,
					 const std::vector<double> &x) {
  std::vector<double> y(a.size(), 0);

  for (size_t i = 0; i < a.size(); ++i) {
    for (size_t j = 0; j < a[0].size(); ++j) {
      y[i] += a[i][j] * x[j];
    }
  }

  return y;
}

// TODO: remove this method
static void verifyEqual(const std::vector<double> &a, const std::vector<double> &b) {
  if (a.size() != b.size())
    fatal("Verification failed!");
  
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::fabs(a[i] - b[i]) > 0.000000000001) {
      fatal("Verification failed");
    }
  }
}


// Read a sparse matrix from an input file in Matrix Market format.
static SparseMatrix createSparseMatrixFromMMFile(const std::string& input_filepath) {
  std::ifstream fstr;
  fstr.open(input_filepath);
  if (fstr.fail())
    fatal("Error opening the matrix input file.");

  // For simplicity, we first construct a dense representation.
  std::vector<std::vector<double>> mat_dense;
  bool size_line_consumed = false;
  size_t rows, cols, nnz;
  for (std::string line; std::getline(fstr, line); ) {
    if (!line.empty() && line[0] != '%') {
      std::istringstream iss{line};
      if (!size_line_consumed) {
	iss >> rows;
	iss >> cols;
	iss >> nnz;
	size_line_consumed = true;
	mat_dense.resize(rows);
	for (std::vector<double> &row : mat_dense)
	  row.resize(cols, std::numeric_limits<double>::quiet_NaN());
      } else {
	size_t row, col;
	iss >> row;
	iss >> col;
	mat_dense[row - 1][col - 1] = (static_cast<double>(rand()) / RAND_MAX) * 2 - 1;
      }
    }
  }
  if (fstr.bad())
    fatal("Error reading from the input file");

  // Create a SparseMatrix object from the dense representation.
  SparseMatrix mat{rows, cols, nnz};
  mat.colptrs_[0] = 0;
  size_t idx = 0;
  for (size_t j = 0; j < cols; ++j) {
    for (size_t i = 0; i < rows; ++i) {
      if (!std::isnan(mat_dense[i][j])) {
	mat.nz_values_[idx] = mat_dense[i][j];
	mat.rowids_[idx] = static_cast<idx_t>(i);
	++idx;
      }
    }
    mat.colptrs_[j + 1] = idx;
  }
  if (idx != mat.nnz_)    
    fatal("Malformed input file");

  return mat;
}

// Create a random sparse vector of the specified length with the specified number of non-zero values.
static SparseVector createRandomSparseVector(size_t length, size_t nnz) {
  SparseVector v{length};
  v.nnz_ = nnz;
  
  std::unordered_set<size_t> nz_indices;
  for (size_t i = 0; i < nnz; ++i) {
    while (true) {
      idx_t idx = static_cast<idx_t>(static_cast<double>(rand()) / RAND_MAX * static_cast<double>(length));
      if (nz_indices.insert(idx).second) {
	v.nz_values_[i] = {idx, static_cast<double>(rand()) / RAND_MAX};
	break;
      }
    }
  }

  return v;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Run the SpMSpV performance evaluator.\n";
    std::cerr << "\n";
    std::cerr << "Usage: " << argv[0] << " <input_matrix_filepath> <x_nz_frac>\n";
    std::cerr << "  <input_matrix_filepath>   Input matrix file in MM (Matrix Market) format.\n";
    std::cerr << "  <x_nz_frac>               The fraction of non-zero entries in the randomly-generated input vector (between 0 and 1).\n";
    exit(EXIT_FAILURE);
  }
  
  // Initialize the sparse matrix from the input file.
  std::cout << "Reading the input matrix from " << argv[1] << ".\n";
  SparseMatrix a = createSparseMatrixFromMMFile(argv[1]);

  double x_nz_frac = std::stod(argv[2]);
  if (x_nz_frac < 0 || x_nz_frac > 1)
    fatal("Invalid <x_nz_frac> argument (must be between 0 and 1).\n");

  size_t x_nnz = static_cast<size_t>(a.cols_ * x_nz_frac);  
  std::cout << "Creating the random input vector with " << x_nnz << " entries.\n";
  SparseVector x = createRandomSparseVector(a.cols_, x_nnz);

  SparseVector y(a.rows_);
  SpMSpV spmspv(a.rows_, a.cols_);

  constexpr int NUM_ITERATIONS = 1000;
  std::cout << "Executing " << NUM_ITERATIONS << " iterations of SpMSpV\n";

  size_t result_sink = 0;
  std::chrono::system_clock::time_point t_start = std::chrono::system_clock::now();
  for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
    spmspv.multiply(a, x, &y);
    verifyEqual(sparseVectorToDense(y), denseMultiply(sparseMatrixToDense(a), sparseVectorToDense(x)));
    result_sink += y.nnz_;
  }
  std::chrono::system_clock::time_point t_end = std::chrono::system_clock::now();
  std::cout << "Average runtime: " <<
    (std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / NUM_ITERATIONS) << "us.\n";
  
  std::cout << result_sink << "\n";
 
  return EXIT_SUCCESS;
}
