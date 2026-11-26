### Ambiente
```
data:      2026-11-26T10:11:18-05:00
host:      compute24
versao:    gpu-v1.1-residente
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
| Tempo de treino (cronômetro interno) | 2.2544 | 0.0041 | 2.2497 | 2.2542 | 2.2611 |
| Tempo de leitura dos arquivos | 0.0076 | 0.0006 | 0.0070 | 0.0072 | 0.0086 |
| Tempo de avaliação (teste) | 0.0009 | 0.0000 | 0.0009 | 0.0009 | 0.0009 |
| Tempo total do processo | 5.1468 | 0.0163 | 5.1101 | 5.1514 | 5.1635 |
