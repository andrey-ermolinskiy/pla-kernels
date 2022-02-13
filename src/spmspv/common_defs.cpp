#include "common_defs.h"

#include <iostream>
#include <cstdlib>

void fatal(const std::string &errmsg) {
  std::cerr << "Fatal error: " << errmsg << '\n';
  exit(EXIT_FAILURE);
}
