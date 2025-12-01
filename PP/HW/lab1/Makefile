CXX = mpicxx
CXXFLAGS = -std=c++17 -O3 -fopenmp
TARGET = lab1
SRC = lab1.cc

.PHONY: all
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f $(TARGET)
