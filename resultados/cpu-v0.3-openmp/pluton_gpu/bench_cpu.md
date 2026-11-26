### Ambiente
```
data:      2026-11-26T09:30:56-05:00
host:      compute20
versao:    cpu-v0.3-openmp
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
| Tempo de treino (cronômetro interno) | 0.1316 | 0.0016 | 0.1307 | 0.1311 | 0.1360 |
| Tempo de leitura dos arquivos | 0.0037 | 0.0001 | 0.0036 | 0.0037 | 0.0039 |
| Tempo de avaliação (teste) | 0.0005 | 0.0000 | 0.0005 | 0.0005 | 0.0005 |
| Tempo total do processo | 0.1394 | 0.0016 | 0.1384 | 0.1388 | 0.1435 |
