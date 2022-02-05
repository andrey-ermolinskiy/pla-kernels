#pragma once

#include <cstdlib>
#include <string>
#include <iostream>

// Matrix or vector element index.
typedef uint32_t idx_t;

// Print an error message and terminate the process with a non-zero exit code.
static void fatal(const std::string &errmsg) {
  std::cerr << "Fatal error: " << errmsg << '\n';
  exit(EXIT_FAILURE);
}
