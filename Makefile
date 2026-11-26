# Barenn - build, validação e benchmark.
#   make               compila a versão CPU (./barenn), com OpenMP
#   make OMP_TREINO=1  experimento: OpenMP também dentro do laço de treino
#   make validar       compila e valida a corretude contra o legado
#   make bench         compila e mede (10 repetições)
#   make clean
# -O3 liga a vetorização automática do GCC (padrão das aulas). Sozinho não ganhou
# nada na Franky (Issue #5); o ganho veio do layout contíguo dos pesos (Issue #7).
# Não usamos -Ofast/-ffast-math: eles reordenam somas de ponto flutuante e o
# resultado deixa de ser bit a bit igual ao do legado (medido na Issue #5).
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -fopenmp -Wall -Wextra
ifeq ($(OMP_TREINO),1)
CXXFLAGS += -DBARENN_OMP_TREINO
endif

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
