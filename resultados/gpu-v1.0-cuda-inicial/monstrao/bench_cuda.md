### Ambiente
```
data:      2026-11-26T09:58:30-05:00
host:      compute24
versao:    gpu-v1.0-cuda-inicial
binario:   ./barenn_cuda
cpu:       Intel(R) Xeon(R) Gold 5215 CPU @ 2.50GHz (40 threads lógicas)
kernel:    Linux 5.14.0-503.33.1.el9_5.x86_64
compilador: g++ (GCC) 14.2.0
nvcc:      Build cuda_12.8.r12.8/compiler.35583870_0
gpu:       Tesla V100-PCIE-32GB, 570.124.06, 32768 MiB
OMP_NUM_THREADS=<nao definido>
entrada:   optdigits.tra (treino) / optdigits.tes (teste), 10 repetições + 1 aquecimento
```

### Resultado (10 repetições)
| Métrica | Média (s) | Desvio (s) | Mín (s) | Mediana (s) | Máx (s) |
|:--|--:|--:|--:|--:|--:|
| Tempo de treino (cronômetro interno) | 8.4945 | 0.0362 | 8.4356 | 8.4983 | 8.5492 |
| Tempo de leitura dos arquivos | 0.0075 | 0.0005 | 0.0068 | 0.0077 | 0.0081 |
| Tempo de avaliação (teste) | 0.0453 | 0.0003 | 0.0449 | 0.0452 | 0.0456 |
| Tempo total do processo | 11.4334 | 0.0252 | 11.3911 | 11.4296 | 11.4663 |
