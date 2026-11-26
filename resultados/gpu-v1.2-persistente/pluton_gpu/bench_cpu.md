### Ambiente
```
data:      2026-11-26T10:16:57-05:00
host:      compute20
versao:    gpu-v1.2-persistente
binario:   ./barenn
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
| Tempo de treino (cronômetro interno) | 0.1315 | 0.0014 | 0.1305 | 0.1310 | 0.1355 |
| Tempo de leitura dos arquivos | 0.0039 | 0.0001 | 0.0037 | 0.0039 | 0.0040 |
| Tempo de avaliação (teste) | 0.0005 | 0.0000 | 0.0005 | 0.0005 | 0.0005 |
| Tempo total do processo | 0.1398 | 0.0017 | 0.1388 | 0.1392 | 0.1446 |
