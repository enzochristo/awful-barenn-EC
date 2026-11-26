### Ambiente
```
data:      2026-11-26T09:10:59-05:00
host:      compute24
versao:    cpu-v0.2-flags
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
| Tempo de treino (cronômetro interno) | 0.9212 | 0.0114 | 0.9133 | 0.9181 | 0.9517 |
| Tempo de leitura dos arquivos | 0.0506 | 0.0031 | 0.0480 | 0.0485 | 0.0555 |
| Tempo de avaliação (teste) | 0.0041 | 0.0000 | 0.0041 | 0.0041 | 0.0042 |
| Tempo total do processo | 0.9798 | 0.0115 | 0.9696 | 0.9765 | 1.0081 |
