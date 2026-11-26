### Ambiente
```
data:      2026-11-26T08:15:40-05:00
host:      compute24
versao:    v0.0-baseline
binario:   ./barenn
cpu:       Intel(R) Xeon(R) Gold 5215 CPU @ 2.50GHz (40 threads lógicas)
kernel:    Linux 5.14.0-503.33.1.el9_5.x86_64
compilador: g++ (GCC) 14.2.0
gpu:       Tesla V100-PCIE-32GB, 570.124.06, 32768 MiB
OMP_NUM_THREADS=<nao definido>
entrada:   optdigits.tra (treino) / optdigits.tes (teste), 10 repetições + 1 aquecimento
```

### Resultado (10 repetições)
| Métrica | Média (s) | Desvio (s) | Mín (s) | Mediana (s) | Máx (s) |
|:--|--:|--:|--:|--:|--:|
| Tempo de treino (cronômetro interno) | 1.5901 | 0.0242 | 1.5340 | 1.5948 | 1.6135 |
| Tempo de leitura dos arquivos | 0.0490 | 0.0016 | 0.0476 | 0.0480 | 0.0519 |
| Tempo de avaliação (teste) | 0.0041 | 0.0001 | 0.0041 | 0.0041 | 0.0043 |
| Tempo total do processo | 1.6473 | 0.0236 | 1.5932 | 1.6542 | 1.6689 |
