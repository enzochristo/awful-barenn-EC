# Barenn: relatório de otimização (CPU → CUDA)

> **Sumário executivo.** O legado (`barenn.cpp`, SGD por amostra numa rede 64→32→16→10) foi otimizado em 9 blocos lógicos de experimento, cada um com Issue → branch → commits → PR → merge → tag anotada (o bloco 3 teve a hipótese refutada e ficou sem tag, como manda este enunciado), todos medidos nas duas filas da Franky e **validados bit a bit contra o legado** (perdas das 51 épocas em hexfloat + hash dos pesos, `scripts/validar.sh`).
> * **CPU:** o treino ficou **3,31x** (`pluton_gpu`) e **4,53x** (`monstrao`) mais rápido. O ganho veio de eliminar cópias, do layout contíguo + SIMD e de OpenMP na leitura.
> * **GPU:** o treino ficou **13,8x** e **15,6x** mais rápido que a primeira versão CUDA. Mesmo assim, **a GPU não supera a CPU otimizada** (3,3x e 1,55x mais lenta), porque o SGD por amostra é serial: cada amostra depende dos pesos da anterior. Ir além exigiria mudar o algoritmo (mini-batch), o que mudaria o resultado.
> * **Versão recomendada:** a de CPU, `cpu-v0.3-openmp` (o `./barenn` de qualquer tag posterior). A melhor de GPU é a `gpu-v1.3-streams`.

## Tabela consolidada (todas as versões, duas filas)

Tempo de **treino**, **mediana de 10 repetições** (+1 aquecimento), em `resultados/<tag>/<fila>/bench_*.md`. O speedup é sempre contra o legado da mesma fila. A mediana é usada porque o `pluton_gpu` às vezes mostra dois grupos de tempo; média, desvio, mínimo e máximo estão em cada `bench_*.md`.

| # | Versão (tag) | Técnica | `pluton_gpu` treino | speedup | `monstrao` treino | speedup | Validação |
|:-:|:--|:--|--:|--:|--:|--:|:--|
| 1 | **Baseline legado** ([`v0.0-baseline`](https://github.com/enzochristo/awful-barenn-EC/tree/v0.0-baseline)) | código original, `-O2` | 0,4337 s | 1,00x | 1,5948 s | 1,00x | referência |
| 2 | [`cpu-v0.1-sem-copias`](https://github.com/enzochristo/awful-barenn-EC/tree/cpu-v0.1-sem-copias) | passagem por referência, buffers reutilizados, sem gradiente morto | 0,3130 s | 1,39x | 0,9143 s | 1,74x | BIT-EXATO |
| 3 | *sem tag* (Issue #5, refutada) | `-O3` sozinho | 0,3143 s | 1,38x | 0,9181 s | 1,74x | BIT-EXATO |
| 4 | [`cpu-v0.2-layout-simd`](https://github.com/enzochristo/awful-barenn-EC/tree/cpu-v0.2-layout-simd) | pesos contíguos e transpostos (axpy vetorizável) + `-O3` | 0,1502 s | 2,89x | 0,3365 s | 4,74x | BIT-EXATO |
| 5 | **CPU otimizada** ([`cpu-v0.3-openmp`](https://github.com/enzochristo/awful-barenn-EC/tree/cpu-v0.3-openmp)) | OpenMP na conversão dos arquivos | **0,1311 s** | **3,31x** | **0,3522 s** | **4,53x** | BIT-EXATO |
| 6 | **CUDA inicial** ([`gpu-v1.0-cuda-inicial`](https://github.com/enzochristo/awful-barenn-EC/tree/gpu-v1.0-cuda-inicial)) | 12 kernels + 2 `cudaMemcpy` por amostra | 5,9333 s | 0,07x | 8,4983 s | 0,19x | BIT-EXATO |
| 7 | [`gpu-v1.1-residente`](https://github.com/enzochristo/awful-barenn-EC/tree/gpu-v1.1-residente) | dados residentes na GPU + 1 kernel fundido por amostra | 1,3342 s | 0,33x | 2,2542 s | 0,71x | BIT-EXATO |
| 8 | [`gpu-v1.2-persistente`](https://github.com/enzochristo/awful-barenn-EC/tree/gpu-v1.2-persistente) | 1 kernel persistente, parâmetros em shared, prefetch | 0,4536 s | 0,96x | 0,5578 s | 2,86x | BIT-EXATO |
| 9 | **CUDA otimizado** ([`gpu-v1.3-streams`](https://github.com/enzochristo/awful-barenn-EC/tree/gpu-v1.3-streams)) | streams + pinned, inferência em shared, bloco de 512 threads | **0,4299 s** | **1,01x** | **0,5465 s** | **2,92x** | BIT-EXATO (+ blocos 128/256/512) |

**Tempo total do processo** (mediana) das quatro versões pedidas:

| Versão | `pluton_gpu` | `monstrao` | Observação |
|:--|--:|--:|:--|
| Baseline legado | 0,4571 s | 1,6542 s | |
| CPU otimizada (`cpu-v0.3-openmp`) | **0,1388 s (3,29x)** | **0,3656 s (4,52x)** | |
| CUDA inicial (`gpu-v1.0-cuda-inicial`) | 8,0742 s | 11,4296 s | |
| CUDA otimizado (`gpu-v1.3-streams`) | 2,5460 s | 3,4404 s | ≈2,1 s e ≈2,9 s são a inicialização do driver e do contexto CUDA, iguais em todas as versões GPU e fora do código |

**GPU contra CPU na mesma máquina** (treino): a v1.0 era 45x e 24x mais lenta; a v1.3 é 3,3x e 1,55x mais lenta.

**Hardware das filas** (`resultados/<tag>/<fila>/ambiente.txt` e `gpu_info.txt`):

| Fila | CPU (o que o job vê) | GPU | Software |
|:--|:--|:--|:--|
| `pluton_gpu` (compute20) | Intel Core i9-14900 (8 CPUs no job) | NVIDIA RTX A1000 8 GB, sm_86, 18 SMs, PCIe 4.0 | g++ 14.2.0, CUDA 12.8, driver 570.124 |
| `monstrao` (compute24) | 2× Intel Xeon Gold 5215 (40 threads) | NVIDIA Tesla V100-PCIE 32 GB, sm_70, 80 SMs, PCIe 3.0 | g++ 14.2.0, CUDA 12.8, driver 570.124 |

## Índice navegável (Issue com a discussão completa → PR → tag)

| Bloco | Issue (hipótese no corpo, resposta nos comentários) | PR | Tag | Resultado em uma linha |
|:-:|:--|:-:|:--|:--|
| 1 | [#1 Baseline reprodutível](https://github.com/enzochristo/awful-barenn-EC/issues/1) | [#2](https://github.com/enzochristo/awful-barenn-EC/pull/2) | [`v0.0-baseline`](https://github.com/enzochristo/awful-barenn-EC/tree/v0.0-baseline) | build, validação bit a bit, bench e callgrind (Backward 67,65% das instruções) |
| 2 | [#3 Cópias e alocações](https://github.com/enzochristo/awful-barenn-EC/issues/3) | [#4](https://github.com/enzochristo/awful-barenn-EC/pull/4) | [`cpu-v0.1-sem-copias`](https://github.com/enzochristo/awful-barenn-EC/tree/cpu-v0.1-sem-copias) | −86% de alocações, 1,39x / 1,74x |
| 3 | [#5 `-O3` (e `-Ofast`)](https://github.com/enzochristo/awful-barenn-EC/issues/5) | [#6](https://github.com/enzochristo/awful-barenn-EC/pull/6) (fechado) · [#19](https://github.com/enzochristo/awful-barenn-EC/pull/19) (dados) | **sem tag** | **refutada**: +0,4% no g++ 14.2; `-Ofast` reprovado na validação |
| 4 | [#7 Layout contíguo + SIMD](https://github.com/enzochristo/awful-barenn-EC/issues/7) | [#8](https://github.com/enzochristo/awful-barenn-EC/pull/8) | [`cpu-v0.2-layout-simd`](https://github.com/enzochristo/awful-barenn-EC/tree/cpu-v0.2-layout-simd) | 2,08x / 2,72x sobre o bloco 2 |
| 5 | [#9 OpenMP](https://github.com/enzochristo/awful-barenn-EC/issues/9) | [#10](https://github.com/enzochristo/awful-barenn-EC/pull/10) | [`cpu-v0.3-openmp`](https://github.com/enzochristo/awful-barenn-EC/tree/cpu-v0.3-openmp) | leitura 5,6x / 7,4x; total ≈1,11x (teto de Amdahl 1,14x); treino em paralelo 2,2x a 7,1x **mais lento** |
| 6 | [#11 CUDA inicial](https://github.com/enzochristo/awful-barenn-EC/issues/11) | [#12](https://github.com/enzochristo/awful-barenn-EC/pull/12) | [`gpu-v1.0-cuda-inicial`](https://github.com/enzochristo/awful-barenn-EC/tree/gpu-v1.0-cuda-inicial) | correto, mas 45x / 24x mais lento que a CPU (nsys: 127.791 kernels de 1,5 µs) |
| 7 | [#13 Residente + kernel fundido](https://github.com/enzochristo/awful-barenn-EC/issues/13) | [#14](https://github.com/enzochristo/awful-barenn-EC/pull/14) | [`gpu-v1.1-residente`](https://github.com/enzochristo/awful-barenn-EC/tree/gpu-v1.1-residente) | 4,45x / 3,77x sobre a v1.0 |
| 8 | [#15 Persistente + shared + prefetch](https://github.com/enzochristo/awful-barenn-EC/issues/15) | [#16](https://github.com/enzochristo/awful-barenn-EC/pull/16) | [`gpu-v1.2-persistente`](https://github.com/enzochristo/awful-barenn-EC/tree/gpu-v1.2-persistente) | 2,94x / 4,04x sobre a v1.1 |
| 9 | [#17 Block size + streams + pinned](https://github.com/enzochristo/awful-barenn-EC/issues/17) | [#18](https://github.com/enzochristo/awful-barenn-EC/pull/18) | [`gpu-v1.3-streams`](https://github.com/enzochristo/awful-barenn-EC/tree/gpu-v1.3-streams) | bloco 512: 1,055x / 1,021x; streams sem efeito mensurável |
| 10 | [#20 Este README](https://github.com/enzochristo/awful-barenn-EC/issues/20) | — | `v2.0-entrega` | consolidação |

## Como reproduzir qualquer versão
```bash
git checkout <tag>
make && make validar                     # CPU: compila e confere bit a bit com o legado
scripts/bench.sh ./barenn 10             # 1 aquecimento + 10 repetições
module load cuda/12.8.1                  # versões gpu-*: na Franky
make cuda && make validar-cuda && scripts/bench.sh ./barenn_cuda 10
FILAS="pluton_gpu monstrao" VERSAO=<tag> scripts/franky.sh   # medição oficial nas duas filas
```
O `franky.sh` compila no nó de login (padrão das aulas) e submete um job por fila com `--exclusive`, que valida, mede e faz o profiling de GPU com `nsys`. O `callgrind` dos blocos de CPU foi feito no laptop (g++ 11.4), porque os nós não têm `valgrind`. Os resultados ficam em `resultados/<tag>/<fila>/` e os logs do SLURM em `outputs/<tag>/<fila>/`.

## O que limitou cada etapa (resumo da discussão)
* **CPU:**
  * o ganho grande veio de deixar o laço vetorizável sem reordenar nenhuma soma (bloco 4). A redução do produto escalar virou axpy, o que mantém a saída bit a bit igual.
  * `-Ofast` seria mais rápido, mas reprova a validação.
  * O OpenMP só ajuda na leitura: o treino é 94% a 96% do tempo e é serial (Lei de Amdahl).
* **GPU:**
  * a v1.0 era limitada por latência de lançamento e cópias síncronas;
  * a v1.1, pela duração de um kernel por amostra;
  * a v1.2 e a v1.3, pela cadeia de 10 barreiras por amostra dentro de 1 SM, com 1 de 18 SMs (A1000) e 1 de 80 (V100) em uso.
  * Na v1.1 o tempo escalou com o clock dos SMs (V100/A1000 = 1,50x, clocks 1,52x), não com o número de SMs nem com a banda.
* **Limitações das ferramentas:** a Franky não tem `perf` nem `valgrind` nos nós (o callgrind rodou no laptop), e o `ncu` é recusado nas duas GPUs (`ERR_NVGPUCTRPERM`, contadores de desempenho restritos ao administrador). Por isso a análise de GPU usa o `nsys`, os tempos e a configuração de lançamento.

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
