CC = gcc
NVCC = nvcc
HIPCC = hipcc

CFLAGS = -Wall -g
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS = -lm
HIPCCFLAGS = -std=c++14 -O3 --offload-arch=gfx908

TARGETS = seq-attention seq-flashattention hw4

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean

seq-attention: seq-attention.c
	$(CC) $(CFLAGS) -o $@ $? $(LDFLAGS)

seq-flashattention: seq-flashattention.c
	$(CC) $(CFLAGS) -o $@ $? $(LDFLAGS)

hw4: hw4.cu
	$(NVCC) $(NVFLAGS) -o $@ $? $(LDFLAGS)

hw4-amd: hw4.hip
	$(HIPCC) $(HIPCCFLAGS) -o $@ $< $(LDFLAGS)