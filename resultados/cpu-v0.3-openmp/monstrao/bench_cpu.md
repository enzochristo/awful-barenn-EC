### Ambiente
```
data:      2026-11-26T09:30:57-05:00
host:      compute24
versao:    cpu-v0.3-openmp
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
| Tempo de treino (cronômetro interno) | 0.3519 | 0.0010 | 0.3496 | 0.3522 | 0.3531 |
| Tempo de leitura dos arquivos | 0.0067 | 0.0002 | 0.0064 | 0.0067 | 0.0069 |
| Tempo de avaliação (teste) | 0.0012 | 0.0000 | 0.0011 | 0.0012 | 0.0012 |
| Tempo total do processo | 0.3653 | 0.0009 | 0.3633 | 0.3656 | 0.3666 |
