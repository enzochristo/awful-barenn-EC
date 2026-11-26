### Ambiente
```
data:      2026-11-26T10:33:22-05:00
host:      compute24
versao:    gpu-v1.3-streams
binario:   ./barenn
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
| Tempo de treino (cronômetro interno) | 0.3524 | 0.0004 | 0.3520 | 0.3523 | 0.3534 |
| Tempo de leitura dos arquivos | 0.0068 | 0.0003 | 0.0066 | 0.0068 | 0.0075 |
| Tempo de avaliação (teste) | 0.0012 | 0.0000 | 0.0011 | 0.0012 | 0.0012 |
| Tempo total do processo | 0.3660 | 0.0006 | 0.3653 | 0.3661 | 0.3670 |
