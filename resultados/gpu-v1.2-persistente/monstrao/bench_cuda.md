### Ambiente
```
data:      2026-11-26T10:17:15-05:00
host:      compute24
versao:    gpu-v1.2-persistente
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
| Tempo de treino (cronômetro interno) | 0.5578 | 0.0005 | 0.5569 | 0.5578 | 0.5588 |
| Tempo de leitura dos arquivos | 0.0073 | 0.0005 | 0.0069 | 0.0072 | 0.0087 |
| Tempo de avaliação (teste) | 0.0009 | 0.0001 | 0.0009 | 0.0009 | 0.0011 |
| Tempo total do processo | 3.4402 | 0.0146 | 3.4207 | 3.4340 | 3.4652 |
