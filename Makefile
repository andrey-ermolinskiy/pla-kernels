CXX=g++
CPPFLAGS=-g -Wall -std=c++17 -fopenmp

SRCS=evaluate_spmspv_perf.cpp SpMSpV.cpp SparseVector.cpp SparseMatrix.cpp common_defs.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: evaluate_spmspv_perf

evaluate_spmspv_perf: $(OBJS)
	$(CXX) $(CPPFLAGS) -o evaluate_spmspv_perf $(OBJS)

evaluate_spmspv_perf.o: evaluate_spmspv_perf.cpp common_defs.h SparseMatrix.h SparseVector.h SpMSpV.h
SpMSpV.o: SpMSpV.cpp common_defs.h SparseMatrix.h SparseVector.h SpMSpV.h
SparseMatrix.o: SparseMatrix.cpp common_defs.h SparseMatrix.h SparseVector.h
SparseVector.o: SparseVector.cpp SparseVector.h common_defs.h
common_defs.o: common_defs.cpp common_defs.h

clean:
	rm -f $(OBJS) evaluate_spmspv_perf
