#include <vector>
#include <cmath>
#include <chrono>
#include <map>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unordered_set>

#include "SparseMatrix.h"
#include "SparseVector.h"
#include "SpMSpV.h"

// Read a sparse matrix from an input file in Matrix Market format.
static SparseMatrix createSparseMatrixFromMMFile(const std::string& input_filepath) {
  std::ifstream fstr;
  fstr.open(input_filepath);
  if (fstr.fail())
    fatal("Error opening the matrix input file.");

  std::vector<std::map<size_t, double>> input_mat;
  // mat[j] stores a collection of <i -> value> mappings.
  bool size_line_consumed = false;
  size_t rows, cols, nnz;
  srand(0);
  for (std::string line; std::getline(fstr, line); ) {
    if (!line.empty() && line[0] != '%') {
      std::istringstream iss{line};
      if (!size_line_consumed) {
        iss >> rows;
        iss >> cols;
        iss >> nnz;
        size_line_consumed = true;
        input_mat.resize(cols);
      } else {
        size_t row, col;
        iss >> row;
        iss >> col;
        input_mat[col - 1].emplace(row - 1, (static_cast<double>(rand()) / RAND_MAX) * 2 - 1);
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
    for (auto [i, value]: input_mat[j]) {
      mat.nz_values_[idx] = value;
      mat.rowids_[idx] = static_cast<idx_t>(i);
      ++idx;
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

  size_t x_nnz = static_cast<size_t>(static_cast<double>(a.cols_) * x_nz_frac);
  std::cout << "Creating a random input vector with " << x_nnz << " non-zero entries.\n";
  SparseVector x = createRandomSparseVector(a.cols_, x_nnz);

  SparseVector y(a.rows_);
  SpMSpV spmspv(a.rows_, a.cols_);

  constexpr int NUM_ITERATIONS = 1000;
  std::cout << "Executing " << NUM_ITERATIONS << " iterations of SpMSpV\n";

  size_t result_sink = 0;
  std::chrono::system_clock::time_point t_start = std::chrono::system_clock::now();
  for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
    spmspv.multiply(a, x, &y);
    result_sink += y.nnz_;
  }
  std::chrono::system_clock::time_point t_end = std::chrono::system_clock::now();
  std::cout << "Average runtime: " <<
    (std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / NUM_ITERATIONS) << "us.\n";

  std::cout << result_sink << "\n";

  return EXIT_SUCCESS;
}
