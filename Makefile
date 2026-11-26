# Barenn - build, validação e benchmark.
#   make               compila a versão CPU (./barenn), com OpenMP
#   make OMP_TREINO=1  experimento: OpenMP também dentro do laço de treino
#   make validar       compila e valida a corretude contra o legado
#   make bench         compila e mede (10 repetições)
#   make cuda          compila a versão GPU (./barenn_cuda); exige nvcc
#   make validar-cuda  compila e valida a versão GPU contra o legado
#   make bench-cuda    compila e mede a versão GPU (10 repetições)
#   make clean
# Na Franky: module load cuda/12.8.1 antes de compilar.
# Sem CUDA_ARCH o nvcc gera PTX, compilado para a GPU do nó na primeira execução
# (funciona em qualquer GPU). Para fixar: make cuda CUDA_ARCH=sm_80.
# BLOCO_TREINO = threads do bloco de treino (padrão 512, o melhor da varredura
# 128/256/512 da Issue #17): make cuda BLOCO_TREINO=256
# --fmad=false impede a contração a*b+c -> FMA na GPU, mantendo o resultado
# bit a bit igual ao da CPU.
# -O3 liga a vetorização automática do GCC (padrão das aulas). Sozinho não ganhou
# nada na Franky (Issue #5); o ganho veio do layout contíguo dos pesos (Issue #7).
# Não usamos -Ofast/-ffast-math: eles reordenam somas de ponto flutuante e o
# resultado deixa de ser bit a bit igual ao do legado (medido na Issue #5).
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -fopenmp -Wall -Wextra
ifeq ($(OMP_TREINO),1)
CXXFLAGS += -DBARENN_OMP_TREINO
endif

NVCC      ?= nvcc
CUDA_ARCH ?=
NVCCFLAGS ?= -std=c++17 -O3 $(if $(CUDA_ARCH),-arch=$(CUDA_ARCH)) --fmad=false -lineinfo \
             -Xcompiler "-O3 -fopenmp -Wall -Wextra"
BLOCO_TREINO ?= 512
NVCCFLAGS += -DBLOCO_TREINO=$(BLOCO_TREINO)

.PHONY: all cuda validar validar-cuda bench bench-cuda clean

all: barenn

barenn: barenn.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

validar: barenn
	./scripts/validar.sh ./barenn

bench: barenn
	./scripts/bench.sh ./barenn 10

cuda: barenn_cuda

barenn_cuda: barenn_cuda.cu FORCE
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lgomp

validar-cuda: barenn_cuda
	./scripts/validar.sh ./barenn_cuda

bench-cuda: barenn_cuda
	./scripts/bench.sh ./barenn_cuda 10

# Recompila sempre: BLOCO_TREINO pode mudar entre chamadas.
FORCE:

clean:
	rm -f barenn barenn_cuda
