### Ambiente
```
data:      2026-11-26T09:21:43-05:00
host:      compute20
versao:    cpu-v0.2-layout-simd
binario:   ./barenn
cpu:       Intel(R) Core(TM) i9-14900 (8 threads lógicas)
kernel:    Linux 5.14.0-503.33.1.el9_5.x86_64
compilador: g++ (GCC) 14.2.0
gpu:       NVIDIA RTX A1000, 570.124.06, 8188 MiB
OMP_NUM_THREADS=<nao definido>
entrada:   optdigits.tra (treino) / optdigits.tes (teste), 10 repetições + 1 aquecimento
```

### Resultado (10 repetições)
| Métrica | Média (s) | Desvio (s) | Mín (s) | Mediana (s) | Máx (s) |
|:--|--:|--:|--:|--:|--:|
| Tempo de treino (cronômetro interno) | 0.1460 | 0.0132 | 0.1312 | 0.1502 | 0.1591 |
| Tempo de leitura dos arquivos | 0.0208 | 0.0018 | 0.0192 | 0.0202 | 0.0235 |
| Tempo de avaliação (teste) | 0.0005 | 0.0001 | 0.0005 | 0.0005 | 0.0006 |
| Tempo total do processo | 0.1703 | 0.0138 | 0.1539 | 0.1749 | 0.1861 |
