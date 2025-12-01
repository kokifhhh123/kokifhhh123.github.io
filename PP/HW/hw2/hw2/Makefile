# ==== Compiler & Common Flags ====
CC      = gcc
CXX     = g++
LDLIBS  = -lpng -lm
OPTFLAGS= -O3 -march=native -funroll-loops -fno-math-errno -fomit-frame-pointer
CFLAGS  = $(OPTFLAGS)
CXXFLAGS= $(CFLAGS)

# ==== Default targets ====
.PHONY: all clean hw2a hw2b
all: hw2a hw2b

# ==== Build hw2a (pthread version) ====
hw2a: hw2a.cc
	$(CXX) $(CXXFLAGS) -pthread $< -o $@ $(LDLIBS)

# ==== Build hw2b (MPI + OpenMP version) ====
hw2b: hw2b.cc
	mpicxx -fopenmp -msse2 $(OPTFLAGS) $< -o $@ $(LDLIBS)

# ==== Clean ====
clean:
	rm -f hw2a hw2b