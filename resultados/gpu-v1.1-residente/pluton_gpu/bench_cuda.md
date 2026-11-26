### Ambiente
```
data:      2026-11-26T10:11:10-05:00
host:      compute20
versao:    gpu-v1.1-residente
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
| Tempo de treino (cronômetro interno) | 1.3339 | 0.0041 | 1.3248 | 1.3342 | 1.3394 |
| Tempo de leitura dos arquivos | 0.0041 | 0.0003 | 0.0038 | 0.0041 | 0.0048 |
| Tempo de avaliação (teste) | 0.0004 | 0.0001 | 0.0004 | 0.0004 | 0.0005 |
| Tempo total do processo | 3.4450 | 0.0080 | 3.4295 | 3.4482 | 3.4553 |
