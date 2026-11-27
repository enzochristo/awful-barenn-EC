# Barenn: análise de desempenho na CPU e na GPU

**Objeto:** a rede neural legada `barenn.cpp`: rede densa 64 → 32 → 16 → 10, SGD por amostra, 51 épocas × 3.823 imagens = **194.973 passos de treino**, avaliação em 1.797 imagens.
**Onde foi medido:** cluster Franky, filas `pluton_gpu` e `monstrao`, nó exclusivo, 1 aquecimento + 10 repetições por versão.
**Critério de corretude:** toda versão reproduz o legado **bit a bit**: a perda de cada época (em hexfloat) e o hash de todos os pesos.

Todos os números abaixo vêm de `resultados/<tag>/<fila>/`. Os tempos são a **mediana** das 10 repetições; média, desvio, mínimo e máximo estão em cada `bench_*.md`.

## 1. Resultado

### 1.1 Tempo de treino por versão
| Versão (tag) | O que mudou | `pluton_gpu` | × legado | `monstrao` | × legado | Corretude |
|:--|:--|--:|--:|--:|--:|:--|
| **Baseline legado** ([`v0.0-baseline`](https://github.com/enzochristo/awful-barenn-EC/tree/v0.0-baseline)) | código original, `-O2` | 0,4337 s | 1,00 | 1,5948 s | 1,00 | referência |
| [`cpu-v0.1-sem-copias`](https://github.com/enzochristo/awful-barenn-EC/tree/cpu-v0.1-sem-copias) | sem cópias e alocações por amostra | 0,3130 s | 1,39 | 0,9143 s | 1,74 | bit a bit |
| (sem tag, [#5](https://github.com/enzochristo/awful-barenn-EC/issues/5)) | `-O3` sozinho | 0,3143 s | 1,38 | 0,9181 s | 1,74 | bit a bit |
| [`cpu-v0.2-layout-simd`](https://github.com/enzochristo/awful-barenn-EC/tree/cpu-v0.2-layout-simd) | pesos contíguos e transpostos + `-O3` | 0,1502 s | 2,89 | 0,3365 s | 4,74 | bit a bit |
| **CPU otimizada** ([`cpu-v0.3-openmp`](https://github.com/enzochristo/awful-barenn-EC/tree/cpu-v0.3-openmp)) | + OpenMP na leitura dos arquivos | **0,1311 s** | **3,31** | **0,3522 s** | **4,53** | bit a bit |
| **CUDA inicial** ([`gpu-v1.0-cuda-inicial`](https://github.com/enzochristo/awful-barenn-EC/tree/gpu-v1.0-cuda-inicial)) | 12 kernels + 2 cópias síncronas por passo | 5,9333 s | 0,07 | 8,4983 s | 0,19 | bit a bit |
| [`gpu-v1.1-residente`](https://github.com/enzochristo/awful-barenn-EC/tree/gpu-v1.1-residente) | dados na GPU, 1 kernel por passo | 1,3342 s | 0,33 | 2,2542 s | 0,71 | bit a bit |
| [`gpu-v1.2-persistente`](https://github.com/enzochristo/awful-barenn-EC/tree/gpu-v1.2-persistente) | 1 kernel para todo o treino, parâmetros em shared | 0,4536 s | 0,96 | 0,5578 s | 2,86 | bit a bit |
| **CUDA otimizado** ([`gpu-v1.3-streams`](https://github.com/enzochristo/awful-barenn-EC/tree/gpu-v1.3-streams)) | + bloco de 512 threads, streams, inferência em shared | **0,4299 s** | **1,01** | **0,5465 s** | **2,92** | bit a bit (blocos 128, 256 e 512) |

### 1.2 Tempo total do processo
Inclui leitura, avaliação e, nas versões GPU, a inicialização do CUDA.

| Versão | `pluton_gpu` | `monstrao` |
|:--|--:|--:|
| Baseline legado | 0,4571 s | 1,6542 s |
| CPU otimizada | 0,1388 s (3,29x mais rápido) | 0,3656 s (4,52x mais rápido) |
| CUDA inicial | 8,0742 s | 11,4296 s |
| CUDA otimizado | 2,5460 s | 3,4404 s |

### 1.3 O que os números mostram
1. **A versão de CPU otimizada é a mais rápida nas duas máquinas.**
   * O passo de treino caiu de 2,22 µs para **0,67 µs** no `pluton_gpu` e de 8,18 µs para **1,81 µs** no `monstrao`.
2. **A GPU ficou 13,8x (`pluton_gpu`) e 15,6x (`monstrao`) mais rápida entre a primeira e a última versão CUDA, mas termina atrás da CPU otimizada da mesma máquina:**
   * 0,4299 s contra 0,1311 s (3,28x) no `pluton_gpu`;
   * 0,5465 s contra 0,3522 s (1,55x) no `monstrao`.

   No `pluton_gpu`, a melhor GPU empata com o código legado em CPU (0,4299 s contra 0,4337 s).
3. **Todo processo que usa a GPU paga ≈2,1 s (`pluton_gpu`) e ≈2,9 s (`monstrao`) de inicialização do driver e do contexto CUDA, fora dos cronômetros.** O valor é o mesmo em todas as versões GPU e não depende do código. Na versão final, ele é 83% e 84% do tempo total (seção 7).
4. **Nenhum ganho foi obtido mudando o resultado.** As duas alternativas mais rápidas que mudam os bits foram medidas e rejeitadas: `-Ofast` e treinar em mini-batch (seção 4.3).

## 2. Índice: Issue (hipótese e discussão), PR e tag de cada etapa
| Etapa | Issue | PR | Tag |
|:-:|:--|:-:|:--|
| 1 | [#1 Baseline reprodutível: build, validação bit a bit, benchmark e profiling](https://github.com/enzochristo/awful-barenn-EC/issues/1) | [#2](https://github.com/enzochristo/awful-barenn-EC/pull/2) | [`v0.0-baseline`](https://github.com/enzochristo/awful-barenn-EC/tree/v0.0-baseline) |
| 2 | [#3 Eliminar cópias e alocações por amostra](https://github.com/enzochristo/awful-barenn-EC/issues/3) | [#4](https://github.com/enzochristo/awful-barenn-EC/pull/4) | [`cpu-v0.1-sem-copias`](https://github.com/enzochristo/awful-barenn-EC/tree/cpu-v0.1-sem-copias) |
| 3 | [#5 `-O3` (e `-Ofast` como controle): hipótese refutada](https://github.com/enzochristo/awful-barenn-EC/issues/5) | [#6](https://github.com/enzochristo/awful-barenn-EC/pull/6), [#19](https://github.com/enzochristo/awful-barenn-EC/pull/19) | sem tag (sem ganho) |
| 4 | [#7 Pesos contíguos e transpostos para vetorizar](https://github.com/enzochristo/awful-barenn-EC/issues/7) | [#8](https://github.com/enzochristo/awful-barenn-EC/pull/8) | [`cpu-v0.2-layout-simd`](https://github.com/enzochristo/awful-barenn-EC/tree/cpu-v0.2-layout-simd) |
| 5 | [#9 OpenMP na leitura e no treino](https://github.com/enzochristo/awful-barenn-EC/issues/9) | [#10](https://github.com/enzochristo/awful-barenn-EC/pull/10) | [`cpu-v0.3-openmp`](https://github.com/enzochristo/awful-barenn-EC/tree/cpu-v0.3-openmp) |
| 6 | [#11 Porting inicial para CUDA](https://github.com/enzochristo/awful-barenn-EC/issues/11) | [#12](https://github.com/enzochristo/awful-barenn-EC/pull/12) | [`gpu-v1.0-cuda-inicial`](https://github.com/enzochristo/awful-barenn-EC/tree/gpu-v1.0-cuda-inicial) |
| 7 | [#13 Dados residentes e kernel fundido](https://github.com/enzochristo/awful-barenn-EC/issues/13) | [#14](https://github.com/enzochristo/awful-barenn-EC/pull/14) | [`gpu-v1.1-residente`](https://github.com/enzochristo/awful-barenn-EC/tree/gpu-v1.1-residente) |
| 8 | [#15 Kernel persistente, shared memory e prefetch](https://github.com/enzochristo/awful-barenn-EC/issues/15) | [#16](https://github.com/enzochristo/awful-barenn-EC/pull/16) | [`gpu-v1.2-persistente`](https://github.com/enzochristo/awful-barenn-EC/tree/gpu-v1.2-persistente) |
| 9 | [#17 Block size, streams e memória pinned](https://github.com/enzochristo/awful-barenn-EC/issues/17) | [#18](https://github.com/enzochristo/awful-barenn-EC/pull/18) | [`gpu-v1.3-streams`](https://github.com/enzochristo/awful-barenn-EC/tree/gpu-v1.3-streams) |
| 10 | [#20 Relatório final](https://github.com/enzochristo/awful-barenn-EC/issues/20) | [#21](https://github.com/enzochristo/awful-barenn-EC/pull/21) | [`v2.0-entrega`](https://github.com/enzochristo/awful-barenn-EC/tree/v2.0-entrega) |

## 3. Onde o tempo estava no legado
Saída do callgrind para o binário da tag `v0.0-baseline` (`resultados/perfil-cpu-laptop/v0.0-baseline_callgrind.txt`; laptop, g++ 11.4, porque os nós da Franky não têm valgrind):
```
14,074,510,163 (100.0%)  PROGRAM TOTALS
9,645,000,172 (68.53%)  ???:Barenn::DenseLayer::Backward(std::vector<float, std::allocator<float> >, float)
3,460,003,728 (24.58%)  ???:Barenn::DenseLayer::Forward(std::vector<float, std::allocator<float> >)
  188,966,340 ( 1.34%)  ./malloc/./malloc/malloc.c:_int_free
  126,664,431 ( 0.90%)  ./malloc/./malloc/malloc.c:malloc
   64,825,121 ( 0.46%)  ./malloc/./malloc/malloc.c:free
```
* Esta saída é uma nova execução, gerada para este relatório. O total coincide com o da [#1](https://github.com/enzochristo/awful-barenn-EC/issues/1) (14,07 G); a divisão entre funções difere em cerca de 1 ponto percentual (lá: `Backward` 67,65% e `Forward` 23,83%).
* **93% das instruções estão no treino** (`Backward` + `Forward`). São ≈67 mil instruções de treino por passo (13,1 G / 194.973) para 6.112 multiplicações-acumulações úteis.
* `Forward` e `Backward` recebem os vetores **por valor** (a assinatura aparece na própria saída): cada passo copia e aloca vetores no heap. O DHAT contou **2.946.598 alocações** ([#3](https://github.com/enzochristo/awful-barenn-EC/issues/3)).
* A leitura dos arquivos era ≈4% do tempo total: 0,0198 s de 0,4571 s no `pluton_gpu` e 0,0480 s de 1,6542 s no `monstrao`.
* O modelo inteiro tem 2.778 parâmetros (≈11 KB) e cabe na cache L1. **O problema não era falta de cache, e sim instruções demais por passo.**

## 4. CPU: o que reduziu o tempo
### 4.1 Contribuição de cada mudança (treino, mediana)
| Mudança | `pluton_gpu` | `monstrao` | Evidência medida | Causa |
|:--|--:|--:|:--|:--|
| Passagem por referência, buffers reutilizados e remoção do gradiente da 1ª camada, que ninguém usava | 0,4337 → 0,3130 s (1,39x) | 1,5948 → 0,9143 s (1,74x) | instruções 14,07 G → 10,26 G (−27%); alocações 2.946.598 → 404.767 (−86%); as restantes vêm todas da leitura | trabalho descartado e tráfego com o alocador saíram do laço |
| Pesos contíguos e transpostos (`wt[j*out+i]`) + `-O3` | 0,3130 → 0,1502 s (2,08x) | 0,9143 → 0,3365 s (2,72x) | instruções 10,26 G → 4,71 G (−54%) | o produto escalar (uma redução, que não vetoriza sem reordenar a soma) virou uma soma elemento a elemento, que vetoriza mantendo a ordem de cada soma |
| OpenMP na conversão dos arquivos | leitura 0,0208 → 0,0037 s (5,6x) | leitura 0,0496 → 0,0067 s (7,4x) | total 1,11x e 1,12x com o mesmo binário (1 thread contra o melhor número de threads) | só a leitura é paralela: ela era ≈12% do total |

Saída do callgrind da versão final (`resultados/perfil-cpu-laptop/cpu-v0.3-openmp_callgrind.txt`):
```
4,708,936,760 (100.0%)  PROGRAM TOTALS
4,256,872,447 (90.40%)  ???:main
   64,486,680 ( 1.37%)  ./stdlib/./stdlib/strtod_l.c:____strtof_l_internal
   24,831,992 ( 0.53%)  ./malloc/./malloc/malloc.c:_int_free
   16,209,209 ( 0.34%)  ./malloc/./malloc/malloc.c:malloc
```
Com `-O3`, o treino é todo embutido em `main` (90,4%). `malloc` + `free` caíram de 2,7% para 0,9% das instruções.

### 4.2 Limite da CPU: Lei de Amdahl
| | `pluton_gpu` | `monstrao` |
|:--|--:|--:|
| Fração do total que é leitura (1 thread) | 12,4% | 12,1% |
| Teto do ganho total paralelizando só a leitura, 1 / (1 − f) | 1,14x | 1,14x |
| Ganho total medido | 1,11x (8 threads) | 1,12x (16 threads) |
| Fração do total que é treino (versão final) | 94,5% | 96,3% |

O treino é serial por definição do algoritmo: o passo *n+1* usa os pesos que o passo *n* acabou de atualizar. Paralelizar dentro do passo foi medido e é **2,2x a 7,1x mais lento** (laptop, 1 a 8 threads, [#9](https://github.com/enzochristo/awful-barenn-EC/issues/9)). Cada região paralela custa de 0,25 µs a 1,31 µs, contra ≈0,8 µs de trabalho útil por passo.

### 4.3 Alternativas medidas e rejeitadas
| Alternativa | Resultado | Motivo da rejeição |
|:--|:--|:--|
| `-O3` sozinho, sem mudar o layout | +0,4% nas duas filas (g++ 14.2) | sem ganho; no g++ 11.4 do laptop deu 1,91x, porque lá o `-O2` não vetoriza |
| `-Ofast` | 2,69x sobre o `-O2` (laptop) | **reprova a validação**: reordena somas de float e muda o resultado |
| OpenMP dentro do passo de treino | 2,2x a 7,1x mais lento | custo de sincronização maior que o trabalho |
| Treinar várias amostras em paralelo (mini-batch) | não implementado | muda o algoritmo e o resultado |
| `-march=native` (AVX2/AVX-512) | não usado | o binário é compilado no login e os nós têm CPUs diferentes ("Illegal instruction") |

## 5. GPU: o que limitou cada versão
### 5.1 Evolução medida
| Versão | Tempo por passo (`pluton_gpu` / `monstrao`) | Lançamentos de kernel no treino | Cópias host↔device no laço | Vazão efetiva | O que limitou (nsys) |
|:--|--:|--:|:--|--:|:--|
| v1.0 inicial | 30,4 / 43,6 µs | ≈2,34 milhões (12 por passo) | 2 síncronas por passo | 0,40 / 0,28 GFLOP/s | custo fixo de lançar kernels de 1,1 a 1,9 µs e espera das cópias; GPU ocupada ≈50% do tempo |
| v1.1 residente | 6,84 / 11,56 µs | 194.973 (1 por passo) | nenhuma | 1,79 / 1,06 GFLOP/s | duração do kernel por amostra (8,06 / 12,12 µs); GPU ocupada ≈88% a 91% |
| v1.2 persistente | 2,33 / 2,86 µs | 1 | nenhuma | 5,25 / 4,27 GFLOP/s | 10 barreiras por amostra dentro de um bloco; 99% do tempo de GPU num só kernel |
| v1.3 final | 2,20 / 2,80 µs | 1 | nenhuma | 5,54 / 4,36 GFLOP/s | o mesmo da v1.2; o bloco de 512 threads encurta a fase mais longa |

A vazão efetiva é 6.112 multiplicações-acumulações × 2 FLOP × 194.973 passos / tempo de treino. A CPU otimizada faz **18,18 / 6,77 GFLOP/s**.

### 5.2 Saída do Nsight Systems: primeira e última versão
`pluton_gpu`, 200 amostras × 51 épocas (`resultados/gpu-v1.0-cuda-inicial/pluton_gpu/nsys_stats.txt`):
```
 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):
 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)  Name
     36.6         69457724      35991    1929.9    1760.0      1311      3360        480.0  Barenn::k_forward
     21.7         41265802      30600    1348.6    1312.0      1151      1664        134.9  Barenn::k_atualiza
     18.1         34403251      30600    1124.3    1088.0       960      1472        111.7  Barenn::k_delta
     15.7         29767841      20400    1459.2    1408.0      1279      1920        146.3  Barenn::k_grad_entrada
      7.8         14862456      10200    1457.1    1408.0      1279      1856        143.6  Barenn::k_perda
 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):
     55.1         10011001  11997     834.5     800.0       448     13408        219.1  [CUDA memcpy Device-to-Host]
     44.9          8155350  12003     679.4     672.0       288     24992        256.4  [CUDA memcpy Host-to-Device]
```
Mesmo recorte, versão final (`resultados/gpu-v1.3-streams/pluton_gpu/nsys_stats.txt`):
```
 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):
 Time (%)  Total Time (ns)  Instances   Avg (ns)    Med (ns)   Min (ns)  Max (ns)  StdDev (ns)  Name
     99.9         26419470          1  26419470.0  26419470.0  26419470  26419470          0.0  Barenn::k_treino_persistente
      0.1            23104          1     23104.0     23104.0     23104     23104          0.0  Barenn::k_inferencia
 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):
     82.9            48255      9    5361.7     512.0       384     37727      12269.0  [CUDA memcpy Host-to-Device]
     17.1             9951      2    4975.5    4975.5      3871      6080       1562.0  [CUDA memcpy Device-to-Host]
```
* **v1.0:** 127.791 kernels de 1,1 a 1,9 µs e 24.000 cópias que somam 3,2 MB (≈133 bytes cada). Cada kernel faz algumas dezenas de operações, e o custo fixo de lançá-lo e esperar a cópia domina.
* **v1.3:** 2 kernels e 11 cópias. Quase todo o tempo de GPU está num único kernel, e as cópias somam 58 µs.

### 5.3 Por que a GPU não passa da CPU neste algoritmo
* **Paralelismo disponível:** cada passo depende do anterior, então só há paralelismo **dentro** do passo: 10 a 32 neurônios por camada e até 2.048 pesos.
* **Uso da GPU:** o treino ocupa **1 bloco em 1 SM**, de 18 SMs na A1000 e 80 na V100.
* **Latência:** cada passo é uma cadeia de 10 fases separadas por barreira, ≈0,22 µs por fase na A1000. A CPU faz o passo inteiro em 0,67 µs, sem barreiras e com SIMD.
* **Prova de que a limitação é latência e não vazão:** na v1.1, o kernel da V100 durava 1,50x o da A1000, a mesma razão dos clocks dos SMs (2.100 / 1.380 MHz = 1,52x). A V100 tem 4,4x mais SMs e banda de memória muito maior, e isso não reduziu o tempo.
* **Ganhar paralelismo exigiria processar várias amostras ao mesmo tempo (mini-batch).** Isso muda a sequência de pesos e o resultado final, e foi descartado pelo critério de corretude.

### 5.4 Tamanho do bloco do treino (`varredura_bloco.md`, treino, mediana)
| Threads por bloco | `pluton_gpu` | `monstrao` |
|--:|--:|--:|
| 128 | 0,5532 s (+22,0%) | 0,6432 s (+15,5%) |
| 256 | 0,4534 s | 0,5571 s |
| **512 (padrão final)** | **0,4303 s (−5,1%)** | **0,5464 s (−1,9%)** |

Duas rodadas independentes deram os mesmos valores com diferença de no máximo 0,9 ms. Os três tamanhos são bit a bit iguais ao legado. A fase mais longa do passo é atualizar os 2.048 pesos da 1ª camada: 16, 8 ou 4 pesos por thread. Encurtar essa fase compensa o custo de cada barreira esperar mais warps.

### 5.5 Técnicas com efeito abaixo do ruído
| Técnica | Efeito medido | Por quê |
|:--|:--|:--|
| Streams + memória pinned (cópia do teste em paralelo ao treino) | treino 0,4536 → 0,4534 s com o mesmo bloco de 256 threads | a cópia escondida é de ≈38 µs; alocar a memória pinned custa 0,37 ms e 1,02 ms |
| Inferência com pesos em shared | kernel 204 → 23 µs (A1000) e 237 → 32 µs (V100) | ≈0,2 ms a menos num total de ≈2,5 s |

## 6. Diferenças entre as duas filas
| | `pluton_gpu` (compute20) | `monstrao` (compute24) |
|:--|:--|:--|
| CPU | Intel Core i9-14900 (8 CPUs visíveis no job) | 2× Intel Xeon Gold 5215, 2,5 GHz (máx. 3,4 GHz), 40 threads |
| GPU | RTX A1000 8 GB, sm_86, 18 SMs, 2.100 MHz, PCIe 4.0 | Tesla V100 32 GB, sm_70, 80 SMs, 1.380 MHz, PCIe 3.0 |
| Treino legado / CPU otimizada | 0,4337 s / 0,1311 s | 1,5948 s / 0,3522 s |
| Ganho da CPU otimizada sobre o legado | 3,31x | 4,53x |
| Escala da leitura com OpenMP (mesmo binário, 1 thread → N) | 5,16x com 8 threads | 8,59x com 16 threads; pior com 32 (threads dividindo núcleos) |
| Treino GPU inicial / final | 5,9333 s / 0,4299 s | 8,4983 s / 0,5465 s |
| GPU final contra a CPU otimizada da mesma máquina | 3,28x mais lenta | 1,55x mais lenta |
| Inicialização do CUDA (fora do código) | ≈2,1 s | ≈2,9 s |

* O código de treino usa um núcleo: o `pluton_gpu` é 3,68x mais rápido no legado e 2,69x na versão final.
* O `monstrao` ganhou mais com as otimizações de CPU. Cada instrução removida custava mais no Xeon, mas o mecanismo não foi medido, porque não há `perf` nos nós.
* Na GPU, a A1000 é mais rápida que a V100 (1,27x na versão final) porque o código é limitado por latência e clock, não pelo número de SMs.

## 7. Custo fixo fora do código: inicialização do CUDA
| Versão | `pluton_gpu`: total − (leitura + treino + avaliação) | `monstrao` |
|:--|--:|--:|
| v1.0 | 2,11 s | 2,88 s |
| v1.1 | 2,11 s | 2,89 s |
| v1.2 | 2,10 s | 2,87 s |
| v1.3 | 2,11 s | 2,89 s |

A diferença é a mesma em todas as versões: carga do driver, criação do contexto e encerramento do processo. Por isso a comparação entre versões GPU usa o tempo de treino. Nas versões de CPU esse custo não existe.

## 8. Limitações das medições
* **`perf` e `valgrind` não existem nos nós da Franky.** O callgrind foi rodado no laptop (i9-13900H, g++ 11.4), então as contagens de instruções podem ser diferentes com o g++ 14.2 do cluster. Não há medida de cache misses nem de IPC.
* **O Nsight Compute (`ncu`) é recusado nas duas GPUs** (`ERR_NVGPUCTRPERM`: contadores de desempenho restritos ao administrador). Não há ocupação alcançada, stalls nem bytes por kernel. A análise de GPU usa o nsys, os tempos e a configuração de lançamento.
* **No `pluton_gpu`, os tempos às vezes se dividem em dois grupos** (≈+20%) desde o baseline, sem relação com o código. Por isso este relatório usa a mediana. Alguns comentários das Issues usaram a média; a conferência está na [#20](https://github.com/enzochristo/awful-barenn-EC/issues/20).
* **O nsys roda em um recorte** (200 amostras × 51 épocas) e acrescenta tempo a cada kernel: na v1.0, ≈23% por passo. Os números do profiler valem para proporções, não como tempo absoluto.
* **Os experimentos `-O2/-O3/-Ofast` e OpenMP no treino foram medidos no laptop.** Na Franky foram medidos só o `-O3` e a leitura paralela.

## 9. Como reproduzir
```bash
git checkout <tag>
make && make validar                      # CPU: compila e confere bit a bit com o legado
scripts/bench.sh ./barenn 10              # 1 aquecimento + 10 repetições
module load cuda/12.8.1                   # versões gpu-* (na Franky)
make cuda && make validar-cuda && scripts/bench.sh ./barenn_cuda 10
FILAS="pluton_gpu monstrao" VERSAO=<tag> scripts/franky.sh   # medição oficial nas duas filas
```
O `franky.sh` compila no nó de login e submete um job exclusivo por fila. O job valida, mede, faz o profiling e, nas versões com `BLOCO_TREINO`, a varredura 128/256/512. Os resultados ficam em `resultados/<tag>/<fila>/` e os logs do SLURM em `outputs/<tag>/<fila>/`.

---

# Projeto Barenn: O Legado, a Sobrevivência e o Salto para a GPU

Era uma segunda-feira aparentemente tranquila quando a demanda chegou à sua mesa. A liderança técnica se aproximou com um sorriso contido e soltou aquela frase clássica que faz qualquer engenheiro gelar por dentro:

> "Temos um componente crítico rodando em produção há anos. O desenvolvedor original já saiu da empresa, mas agora o projeto é oficialmente seu."
    
Você procurou documentação, manuais ou notas de arquitetura e não encontrou rigorosamente nada. Restou apenas um código hermético a correr em produção.

Depois de analisar o código entendeu que é a implementação de uma Feedforward Neural Network, mas por que razão reinventar a roda? No passado, exigências estritas de conformidade e restrições de licenciamento impediram o recurso a bibliotecas de mercado. Sem alternativa, a empresa optou por criar a sua própria implementação de raiz. O resultado funcionou durante anos, mas tornou-se um labirinto técnico esquecido.

Com o aumento exponencial do volume de dados, a máquina atual começou a falhar nos prazos operacionais. Antes de qualquer migração complexa, **o seu primeiro desafio é estabilizar o código e espremer todo o desempenho possível da CPU — atacando gargalos, repensando o acesso à memória e explorando o paralelismo nativo.**

Para resolver o problema a longo prazo, a empresa adquiriu servidores blade equipados com GPUs corporativas NVIDIA. O hardware de alto desempenho já está montado no datacenter: **a sua missão final é reescrever os pontos críticos do algoritmo em CUDA e levar o processamento ao nível do paralelismo massivo.**

---

## Diretrizes de Conduta e Governança no Repositório

Neste projeto, o repositório não é apenas um local de entrega de arquivos, mas a **comprovação do seu processo investigativo de engenharia**. Espera-se uma postura estritamente empírica: nenhuma otimização deve ser feita no escuro e nenhuma melhoria de velocidade tem valor se corromper o resultado final do programa.

---

### O Princípio do Bloco Lógico de Trabalho

Você tem autonomia para decidir **o que** otimizar e **em qual ordem**, mas cada entrega deve ser estruturada como um **bloco lógico fechado**.

> **O que é um bloco lógico?**
> É uma unidade completa de intervenção técnica que engloba: **um problema identificado por profiling $\rightarrow$ uma hipótese de melhoria $\rightarrow$ a alteração no código $\rightarrow$ a validação da corretude $\rightarrow$ a medição do ganho real.**

* **Não crie tags por tempo decorrido:** Evite tags como `versao-dia-2` ou commits de "fim de expediente".
* **Não crie tags para alterações inconclusivas:** Se a alteração piorou o desempenho ou quebrou os testes, documente na Issue, reverta ou ajuste antes de formalizar o marco.
* **Toda tag deve ser reprodutível:** O professor ou qualquer colega deve conseguir executar `git checkout <tag>`, compilar o código e obter exatamente o mesmo comportamento e ordem de grandeza de desempenho documentados.

---

### O Fluxo Esperado no GitHub

Para cada intervenção técnica planejada, seu ciclo de trabalho no repositório deve seguir três pilares:

#### 1. Abertura de Issue (A Hipótese e o Ponto de Partida)

Antes de alterar uma única linha de código, abra uma Issue explicando a intenção do bloco lógico:

* **Diagnóstico:** Qual gargalo você encontrou usando ferramentas de análise (profilers)?
* **Hipótese:** Qual conceito técnico você pretende aplicar para resolver esse gargalo?
* **Métrica Base:** Qual é o tempo ou custo de processamento atual dessa rotina?

#### 2. Desenvolvimento Isolado e Commits Semânticos

* Trabalhe em uma *branch* dedicada àquela hipótese.
* Faça commits descritivos focados na causa da mudança (ex: `perf: ...`, `refactor: ...`, `fix: ...`).
* Certifique-se de que a saída do modelo/algoritmo continua correta. Código rápido que calcula resultados errados é considerado código quebrado.

#### 3. Formalização com Tag Anotada e Fechamento da Issue

Quando o bloco estiver concluído e validado, crie uma **Git Tag anotada**:

* A tag marca o estado exato da base de código naquele degrau de maturidade.
* A mensagem da tag deve resumir sucintamente o resultado alcançado.
* Feche a Issue anexando as evidências (dados de tempo, prints do profiler ou tabelas comparando o "antes e depois").

---

### Exemplo Ilustrativo de um Bloco Lógico

*(Este é apenas um exemplo de como organizar a narrativa técnica e as ferramentas; não representa um passo obrigatório ou a solução do seu problema).*

---

#### 1. Cenário: O aluno percebeu lentidão na etapa de processamento sequencial

* **Abertura da Issue #04:**
> **Título:** *Redução de tempo de computação na rotina de cálculo vetorial*
> **Contexto:** Ao inspecionar o código com o profiler, identificou-se que 65% do tempo de execução é consumido em um laço interno que realiza operações matemáticas repetitivas.
> **Métricas atuais:** Tempo total de execução: `14.2s`.
> **Hipótese:** Reestruturar a ordem dos dados para favorecer o uso de registradores e paralelizar o laço com múltiplos núcleos deve reduzir o tempo dessa rotina.


* **Implementação na branch:** `feature/otimizacao-laco-vetorial`
* Commits atômicos ajustando o laço e aplicando a estratégia escolhida.
* Verificação de que o resultado final do programa bate com o original.


* **Criação da Tag:**
```bash
git tag -a v0.3-laco-paralelo -m "Otimizacao do calculo vetorial: tempo reduzido de 14.2s para 4.8s (Speedup 2.95x). Corretude validada."
git push origin v0.3-laco-paralelo

```


* **Fechamento da Issue:**
* O aluno responde na Issue #04 anexando o comparativo de tempo (antes: `14.2s` / depois: `4.8s`), faz o merge da branch e referencia a tag criada:


> *"Otimização implementada e validada na tag `v0.3-laco-paralelo`. Issue fechada pelo PR #05."*


---

# Rubrica de Avaliação (Total: 10,0 Pontos)

<div style="background:rgba(255, 0, 0, 0.65)"> 

> ### ⚠️ CLÁUSULA ELIMINATÓRIA GERAL (CONDIÇÃO DE NOTA ZERO)
> 
> 
> A avaliação adota uma política estrita de controle de qualidade de engenharia:
> **Se em QUALQUER um dos critérios ou subcritérios de avaliação for observado o indicativo de `0,0 ponto`, a ATIVIDADE SERÁ INTEGRALMENTE ZERADA (Nota Final = 0,0).**
> Não é tolerada a ausência completa de qualquer pilar fundamental da atividade (seja por código inoperante/incorreto, falta de governança no Git ou ausência de embasamento empírico nas Issues).

* **Implementação e Desempenho Técnico:** 4,0 pontos
* **Governança, Reprodutibilidade e Tags:** 2,0 pontos
* **Relatório Técnico nas Issues (Análise e Discussão):** 4,0 pontos

</div>

---

### Critério 1: Implementação e Desempenho Técnico (4,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — O código não compila, aborta durante a execução, não é funcional ou gera saídas matematicamente corrompidas.
* **+0,5 ponto** — Executou com sucesso o código legado, mediu o baseline de referência de forma reproduzível e identificou os gargalos iniciais.
* **+1,0 ponto** — Aplicou otimizações eficazes na CPU com ganhos consistentes de velocidade e paralelismo multicore preservando a integridade numérica.
* **+1,0 ponto** — **Migração Funcional para CUDA:** Porting inicial para GPU completo e correto (kernel funcional, alocação e liberação de memória no Device e transferências Host/Device íntegras).
* **+0,75 ponto** — Otimizou a execução em GPU.
* **+0,75 ponto** — Implementou técnicas avançadas em GPU, comprovando ganhos.

---

### Critério 2: Governança, Reprodutibilidade e Tags (2,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Ausência de tags, ou repositório com commit único ("tudo pronto") / commits lineares sem separação lógica, inviabilizando a reprodutibilidade dos marcos intermediários.
* **+0,5 ponto** — Utilizou tags, mas elas marcam datas/horários arbitrários em vez de blocos lógicos fechados de trabalho, ou as tags não compilam de forma independente.
* **+1,2 ponto** — Trabalho estruturado em **blocos lógicos**: cada intervenção possui uma Git Tag anotada associada a uma Issue, permitindo compilar e checar a métrica de cada versão via `git checkout <tag>`.
* **+2,0 pontos** — Reprodutibilidade e histórico impecáveis: commits atômicos com mensagens semânticas (`perf:`, `refactor:`, `test:`), branches limpas integradas via Pull Requests documentados e rastreabilidade total de cada degrau de performance.

---

### Critério 3: Relatório Técnico nas Issues (Análise e Discussão) (4,0 pontos)

#### 3.1 Profiling e Evidências Empíricas (1,5 ponto)

* **0,0 ponto [ZERA A ATIVIDADE]** — Não apresenta dados de medição nas Issues ou baseia-se unicamente em impressões subjetivas sem qualquer dado empírico.
* **+0,5 ponto** — Apresenta apenas tempos brutos de relógio (*wall-clock*) colados em texto corrido, sem dados de profilers ou repetições estatísticas.
* **+1,0 ponto** — Apresenta dados e capturas das ferramentas oficiais (**perf/callgrind** e **Nsight Systems/Compute**), mas faltam informações cruciais de ambiente (flags, parâmetros de teste ou hardware).
* **+1,5 ponto** — Issues contêm relatórios completos de profiling (CPU e GPU), métricas de hardware detalhadas (tempo de kernel vs transferência PCIe, cache misses, ocupação teórica vs real de SMs) e metodologia experimental rigorosa.

#### 3.2 Qualidade da Discussão Técnica e Argumentação de Engenharia (2,5 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Discussão inexistente, Issues fechadas apenas com "ok/concluído" ou explicações inconsistentes/conflitantes com os dados coletados.
* **+1,0 ponto** — Discussão puramente descritiva (lista o que foi alterado nas linhas de código, mas não analisa *por que* houve ou não ganho com base em hardware).
* **+1,8 ponto** — Discute tecnicamente a relação entre o código e o comportamento de hardware (ex: impacto do layout de memória nos caches da CPU, balanceamento de carga entre threads).
* **+2,5 pontos** — Análise técnica aprofundada: correlaciona os resultados com conceitos avançados de arquitetura (roofline model, limites da Lei de Amdahl, latência do PCIe vs throughput do kernel CUDA) e debate criticamente as decisões de engenharia tomadas.

---

## Tabela de Cenários e Balizadores

| Cenário | Situação Técnica no GitHub | Resultado / Nota |
| --- | --- | --- |
| **Desclassificado** | Tirou `0,0` em qualquer critério (ex: código não compila, sem tags, sem dados de profiling nas issues ou issues vazias). | **0,0 (Zerada)** |
| **Abaixo do Esperado** | Código compila e funciona, mas há otimizações superficiais, profiling fraco e tags incompletas. | **0,1 a 5,9** |
| **Linha de Base (Alvo: 6,0)** | CPU razoavelmente otimizada + **Migração simples para CUDA funcional** + Issues estruturadas com diagnóstico e tags atreladas. | **6,0 a 6,5** |
| **Avançado / Destaque** | CPU altamente eficiente + **CUDA com otimizações avançadas (Shared Memory/Streams)** + discussões ricas nas Issues com Nsight/Perf + governança impecável. | **8,5 a 10,0** |

---

## Formato de Entrega

A entrega consiste exclusivamente no **link do repositório no GitHub**.

1. O `README.md` principal deve atuar como o **sumário executivo do relatório**, sendo adiciocionado **NO TOPO DO README**:
* A tabela consolidada final comparando todas as versões (Baseline Legado, CPU Otimizada, CUDA Inicial, CUDA Otimizado);
* Um **índice navegável com links diretos** para cada uma das Issues/Tags que contêm a discussão detalhada;

2. Todas as branches de trabalho devem estar integradas à `main`;
3. Todas as Issues de experimentos devem estar fechadas e associadas à sua respectiva Git Tag anotada.
