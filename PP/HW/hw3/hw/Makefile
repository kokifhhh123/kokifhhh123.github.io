CC       = gcc
CXX      = g++
NVCC     = nvcc
HIPCC    = hipcc

LDFLAGS  = -lm

CFLAGS_OMP = -O3 -fopenmp


NVFLAGS_BASE = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 

NVFLAGS_HW3_3 = -std=c++11 -O3 \
                -Xptxas="-O3,-v,-dlcm=cg" \
                -arch=sm_61 \
                -Xcompiler="-fopenmp"

HIPCCFLAGS = -std=c++11 -O3 --offload-arch=gfx908

EXES = hw3-1 hw3-2 hw3-3

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

hw3-1: hw3-1.cc
	$(CC) $(CFLAGS_OMP) $(LDFLAGS) -o $@ $<

hw3-2: hw3-2.cu
	$(NVCC) $(NVFLAGS_BASE) $(LDFLAGS) -o $@ $?

hw3-2-amd: hw3-2.hip
	$(HIPCC) $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<

hw3-3: hw3-3.cu
	$(NVCC) $(NVFLAGS_HW3_3) $(LDFLAGS) -o $@ $?

hw3-3-amd: hw3-3.hip
	$(HIPCC) $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<