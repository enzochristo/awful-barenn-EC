# Barenn - build, validação e benchmark.
#   make            compila a versão CPU (./barenn)
#   make validar    compila e valida a corretude contra o legado
#   make bench      compila e mede (10 repetições)
#   make clean
# -O3 liga a vetorização automática do GCC (no GCC 11, o -O2 não vetoriza).
# Não usamos -Ofast/-ffast-math: eles reordenam somas de ponto flutuante e o
# resultado deixa de ser bit a bit igual ao do legado (medido na Issue das flags).
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra

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
