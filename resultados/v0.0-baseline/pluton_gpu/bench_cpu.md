### Ambiente
```
data:      2026-11-26T08:15:37-05:00
host:      compute20
versao:    v0.0-baseline
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
| Tempo de treino (cronômetro interno) | 0.4683 | 0.0461 | 0.4309 | 0.4337 | 0.5222 |
| Tempo de leitura dos arquivos | 0.0204 | 0.0018 | 0.0182 | 0.0198 | 0.0231 |
| Tempo de avaliação (teste) | 0.0012 | 0.0001 | 0.0011 | 0.0011 | 0.0013 |
| Tempo total do processo | 0.4933 | 0.0481 | 0.4549 | 0.4571 | 0.5504 |
