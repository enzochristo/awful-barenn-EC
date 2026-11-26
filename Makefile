# Barenn - build, validação e benchmark.
#   make            compila a versão CPU (./barenn)
#   make validar    compila e valida a corretude contra o legado
#   make bench      compila e mede (10 repetições)
#   make clean
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra

.PHONY: all validar bench clean

all: barenn

barenn: barenn.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

validar: barenn
	./scripts/validar.sh ./barenn

bench: barenn
	./scripts/bench.sh ./barenn 10

clean:
	rm -f barenn barenn_cuda
