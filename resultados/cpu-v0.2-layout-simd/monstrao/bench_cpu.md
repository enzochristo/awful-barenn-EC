### Ambiente
```
data:      2026-11-26T09:21:44-05:00
host:      compute24
versao:    cpu-v0.2-layout-simd
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
| Tempo de treino (cronômetro interno) | 0.3367 | 0.0008 | 0.3362 | 0.3365 | 0.3387 |
| Tempo de leitura dos arquivos | 0.0496 | 0.0030 | 0.0466 | 0.0476 | 0.0542 |
| Tempo de avaliação (teste) | 0.0012 | 0.0001 | 0.0011 | 0.0012 | 0.0013 |
| Tempo total do processo | 0.3915 | 0.0033 | 0.3885 | 0.3899 | 0.3977 |
