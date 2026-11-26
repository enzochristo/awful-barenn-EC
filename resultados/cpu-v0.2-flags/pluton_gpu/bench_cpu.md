### Ambiente
```
data:      2026-11-26T09:10:57-05:00
host:      compute20
versao:    cpu-v0.2-flags
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
| Tempo de treino (cronômetro interno) | 0.3388 | 0.0334 | 0.3118 | 0.3143 | 0.3796 |
| Tempo de leitura dos arquivos | 0.0216 | 0.0023 | 0.0192 | 0.0202 | 0.0247 |
| Tempo de avaliação (teste) | 0.0011 | 0.0001 | 0.0010 | 0.0010 | 0.0012 |
| Tempo total do processo | 0.3648 | 0.0350 | 0.3353 | 0.3408 | 0.4086 |
