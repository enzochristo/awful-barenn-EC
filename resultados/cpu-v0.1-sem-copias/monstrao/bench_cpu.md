### Ambiente
```
data:      2026-11-26T08:48:07-05:00
host:      compute24
versao:    cpu-v0.1-sem-copias
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
| Tempo de treino (cronômetro interno) | 0.9164 | 0.0047 | 0.9129 | 0.9143 | 0.9287 |
| Tempo de leitura dos arquivos | 0.0518 | 0.0030 | 0.0479 | 0.0524 | 0.0555 |
| Tempo de avaliação (teste) | 0.0041 | 0.0000 | 0.0041 | 0.0041 | 0.0042 |
| Tempo total do processo | 0.9764 | 0.0049 | 0.9696 | 0.9764 | 0.9854 |
