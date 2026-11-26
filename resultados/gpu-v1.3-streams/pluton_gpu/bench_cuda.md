### Ambiente
```
data:      2026-11-26T10:33:31-05:00
host:      compute20
versao:    gpu-v1.3-streams
binario:   ./barenn_cuda
cpu:       Intel(R) Core(TM) i9-14900 (8 threads lógicas)
kernel:    Linux 5.14.0-503.33.1.el9_5.x86_64
compilador: g++ (GCC) 14.2.0
nvcc:      Build cuda_12.8.r12.8/compiler.35583870_0
gpu:       NVIDIA RTX A1000, 570.124.06, 8188 MiB
OMP_NUM_THREADS=<nao definido>
entrada:   optdigits.tra (treino) / optdigits.tes (teste), 10 repetições + 1 aquecimento
```

### Resultado (10 repetições)
| Métrica | Média (s) | Desvio (s) | Mín (s) | Mediana (s) | Máx (s) |
|:--|--:|--:|--:|--:|--:|
| Tempo de treino (cronômetro interno) | 0.4538 | 0.0003 | 0.4532 | 0.4537 | 0.4543 |
| Tempo de leitura dos arquivos | 0.0040 | 0.0002 | 0.0038 | 0.0040 | 0.0042 |
| Tempo de avaliação (teste) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Tempo total do processo | 2.5568 | 0.0160 | 2.5272 | 2.5584 | 2.5723 |
