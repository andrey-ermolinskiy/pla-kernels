#include "SpMSpV.h"

#include <omp.h>
#include <string.h>
#include <new>
#include <cmath>
#include <tuple>
#include <limits>

SpMSpV::SpMSpV(size_t rows, size_t cols) :
  scaled_values_{new SparseVector::IdxValue[MAX_MATRIX_NNZ]},
  bucket_state_array_{new (std::align_val_t{64})BucketState[NUM_BUCKETS]},
  spa_{new double[rows]} {
    omp_set_num_threads(NUM_THREADS);
}


SpMSpV::~SpMSpV() {
  delete[] scaled_values_;
  delete[] bucket_state_array_;
  delete[] spa_;
}


void SpMSpV::multiply(const SparseMatrix& a, const SparseVector& x, SparseVector* y) {
  if (a.nnz_ > MAX_MATRIX_NNZ)
    fatal("The number of non-zero elements in the matrix exceeds the maximum supported value.");

  // For each bucket b, y_offset_by_bucket_[b] stores the starting offset at which the elements of
  // bucket b will be written to the output vector in step 5.
  alignas(64) size_t y_offset_by_bucket[NUM_BUCKETS];

  // For each thread i and bucket j, bcount[i * NUM_BUCKETS + j] stores the number of entries that thread i
  // will insert into bucket j in parallel Step 1.
  alignas(64) size_t bcount[NUM_THREADS * NUM_BUCKETS];

  // For each thread i and bucket j, thr_bucket_start_offset_[i * NUM_BUCKETS + j] stores the starting
  // position in the scaled_values_ array at which thread i will insert its contribution to bucket j
  // in parallel step 1.
  alignas(64) size_t thr_bucket_start_offset[NUM_THREADS * NUM_BUCKETS];
  
  #pragma omp parallel
  {
    // -------------------------------- Step 0 (Preprocessing) -------------------------------------------
    // In this step, we populate the bcount_ array of length (NUM_THREADS * NUM_BUCKETS).
    // bcount[i * NUM_BUCKETS + j] stores the number of entries that the i-th thread will insert into the j-th
    // bucket in Step 2. This preprocesing step allows to avoid synchronization among threads when populating the
    // buckets in Step 2.
    const int thr_idx = omp_get_thread_num();
    size_t* const bcount_chunk = &bcount[thr_idx * NUM_BUCKETS];
    memset(bcount_chunk, 0, NUM_BUCKETS * sizeof(size_t));

    // Determine the start/end boundaries of the input vector slice that will be assigned to this thread in Step 2.
    const size_t slice_size = (x.nnz_ + NUM_THREADS - 1) / NUM_THREADS;
    const size_t slice_start_pos = slice_size * thr_idx;
    const size_t slice_end_pos = std::min(slice_start_pos + slice_size, x.nnz_);
    
    for (size_t t = slice_start_pos; t < slice_end_pos; ++t) { // For every nonzero entry x[j] in this thread's slice of x
      const idx_t j = x.nz_values_[t].idx_;
      for (size_t p = a.colptrs_[j]; p < a.colptrs_[j + 1]; ++p) { // For every nonzero entry in the j-th column of a	
	const idx_t i = a.rowids_[p];
	const size_t b = (i * NUM_BUCKETS) / a.rows_;   // Determine the destination bucket for a[i, j] * x[j]
	++bcount_chunk[b];
      }
    }
    #pragma omp barrier
    
    // ------------------------ Step 1 (Calculation of bucket boundaries) ------------------------------
    // In this step, we read the contents of the bcount_ array and populate thr_bucket_start_offset.
    // For each thread i and bucket j, thr_bucket_start_offset[i * NUM_BUCKETS + j] stores the starting
    // position in the scaled_values_ array at which thread i will insert its contribution to bucket j
    // in step 2.
    //
    // Additionally, for each bucket j, we populate the start_offset_ and size_ fields of bucket_state_array_[j],
    // with the bucket's starting offset in the scaled_values_ array and its size, respectively.
    #pragma omp single
    {
      size_t offset = 0;
      for (int b = 0; b < NUM_BUCKETS; ++b) {
	bucket_state_array_[b].start_offset_ = offset;
	size_t t = b;
	for (size_t thr_idx = 0; thr_idx < NUM_THREADS; ++thr_idx, t += NUM_BUCKETS) {
	  thr_bucket_start_offset[t] = offset;
	  offset += bcount[t];
	}
	bucket_state_array_[b].size_ = offset - bucket_state_array_[b].start_offset_;
      }
    }

    // ------------- Step 2 (Accumulation of columns of a into buckets) -------------------------
    // In this step, the values of the columns A[:,j] for which x[j] != 0 are extracted and multiplied by
    // the nonzero values of x. The results of this pairwise multiplication operation are stored in buckets
    // together with their row indices. Each bucket corresponds to a subset of consecutive rows of the matrix.
    size_t* const bucket_start_offset = &thr_bucket_start_offset[thr_idx * NUM_BUCKETS];
            
    for (size_t t = slice_start_pos; t < slice_end_pos; ++t) { // For every nonzero entry x[j] in this slice of x      
      const idx_t j = x.nz_values_[t].idx_;
      const double x_j = x.nz_values_[t].value_;
      for (size_t p = a.colptrs_[j]; p < a.colptrs_[j + 1]; ++p) {  // For every nonzero element in a[:,j]
	// Compute the product a[i, j] * x[j] and store the result at the appropriate location in the
	// destination bucket.
	const idx_t i = a.rowids_[p];
	const double a_i_j = a.nz_values_[p];      
	const size_t bucket_idx = (static_cast<size_t>(i) * NUM_BUCKETS) / a.rows_;
	const size_t scaled_values_offset = bucket_start_offset[bucket_idx]++;
	scaled_values_[scaled_values_offset] = {i, a_i_j * x_j};
      }
    }
    #pragma omp barrier

    // --------- Step 3 (Merging of bucket entries and calculation of unique row indices) -------
    // Split the buckets among worker threads. For each bucket, compute the sum of non-zero entries
    // and construct a set of unique row indices.
    #pragma omp for schedule(guided, 2)    
    for (int b = 0; b < NUM_BUCKETS; ++b) {
      BucketState* const bstate = &bucket_state_array_[b];

      bstate->num_uinds_ = 0;
      SparseVector::IdxValue* const bucket = &scaled_values_[bstate->start_offset_];
      for (size_t t = 0; t < bstate->size_; ++t)
	spa_[bucket[t].idx_] = std::numeric_limits<double>::quiet_NaN();
      
      for (size_t t = 0; t < bstate->size_; ++t) {
	idx_t i = bucket[t].idx_;
	if (std::isnan(spa_[i])) {
	  // First observation of this row index; Add it to the unique row index set.
	  bstate->uinds_[bstate->num_uinds_++] = i;
	  spa_[i] = bucket[t].value_;
	} else
	  spa_[i] += bucket[t].value_;
      }
    }

    // ----------------- Step 4 (Calculation of output vector slices) ----------------
    // For each bucket b, compute the starting offset at which the elements of bucket b will be
    // written to the output vector in step 5. This is a simple prefix sum computation.
    #pragma omp single
    {
      size_t offset = 0;
      for (int b = 0; b < NUM_BUCKETS; ++b) {
	y_offset_by_bucket[b] = offset;
	offset += bucket_state_array_[b].num_uinds_;
      }

      // Note that at this program location, offset represents the total number of non-zero elements
      // in the output vector.
      y->nnz_ = offset;
    }
 
    // -------------------- Step 5 (Population of the output vector) ---------------------
    // For each bucket b, transfer the non-zero values from the SPA to the output vector.
    #pragma omp for schedule(guided, 2)
    for (int b = 0; b < NUM_BUCKETS; ++b) {
      SparseVector::IdxValue* const y_nz_value_chunk = &(y->nz_values_[y_offset_by_bucket[b]]);
      BucketState *bstate = &bucket_state_array_[b];    
      for (size_t u = 0; u < bstate->num_uinds_; ++u) {
	const idx_t i = bstate->uinds_[u];
	y_nz_value_chunk[u] = {i, spa_[i]};
      }
    }
  } // end omp parallel
}

