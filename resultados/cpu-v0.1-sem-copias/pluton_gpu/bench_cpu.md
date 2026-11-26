### Ambiente
```
data:      2026-11-26T08:48:06-05:00
host:      compute20
versao:    cpu-v0.1-sem-copias
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
| Tempo de treino (cronômetro interno) | 0.3131 | 0.0010 | 0.3119 | 0.3130 | 0.3154 |
| Tempo de leitura dos arquivos | 0.0210 | 0.0018 | 0.0194 | 0.0202 | 0.0248 |
| Tempo de avaliação (teste) | 0.0010 | 0.0000 | 0.0010 | 0.0010 | 0.0010 |
| Tempo total do processo | 0.3387 | 0.0025 | 0.3359 | 0.3381 | 0.3450 |
