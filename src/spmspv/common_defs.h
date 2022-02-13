#pragma once

#include <string>

// Matrix and vector element index.
typedef uint32_t idx_t;

// Print an error message and terminate the process with a non-zero exit code.
void fatal(const std::string &errmsg);
