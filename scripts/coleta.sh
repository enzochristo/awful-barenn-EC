#!/usr/bin/env bash
# Coleta completa de UMA versão em UM nó: compila no próprio nó, VALIDA, mede e
# faz profiling da CPU e, se houver barenn_cuda.cu e GPU, também da GPU.
# Chamado pelo scripts/franky.sh dentro de cada job SLURM (pode rodar à mão).
#
# Uso: scripts/coleta.sh <pasta_saida>
# Variáveis: REPETICOES (10), VARREDURA_OMP=1 (mede 1..N threads),
#            CUDA_ARCH (vazio = PTX para qualquer GPU), BLOCO_TREINO, SEM_PROFILING=1
#            VARREDURA_BLOCO ("128 256 512": mede os binários barenn_cuda_b<N> do franky.sh)
#
# Compila numa cópia temporária, dentro do próprio nó: cada fila usa o binário
# gerado no seu hardware e dois jobs simultâneos não disputam os mesmos arquivos.
set -u
# Na Franky o CUDA vem do módulo cuda/12.8.1 (como nas aulas).
if ! command -v nvcc >/dev/null 2>&1; then
    set +u   # o lmod.sh usa variáveis não definidas (PBS_NODEFILE)
    [ -f /etc/profile.d/lmod.sh ] && . /etc/profile.d/lmod.sh
    command -v module >/dev/null 2>&1 && module load cuda/12.8.1 >/dev/null 2>&1
    set -u
fi
export LC_ALL=C
RAIZ=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(realpath -m "${1:?pasta de saida}")
mkdir -p "$OUT"
export VERSAO=${VERSAO:-$(git -C "$RAIZ" describe --tags --always --dirty 2>/dev/null || echo atual)}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
(cd "$RAIZ" && tar --exclude=./resultados --exclude=./outputs --exclude=./.git -cf - .) | tar -xf - -C "$WORK"
cd "$WORK" || exit 1
falhou=0

{
    echo "versao:     $VERSAO"
    echo "data:       $(date -Iseconds)"
    echo "no:         $(hostname)"
    echo "fila:       ${SLURM_JOB_PARTITION:-<fora do SLURM>}  job ${SLURM_JOB_ID:-?}"
    echo "cpus job:   ${SLURM_CPUS_ON_NODE:-?}  OMP_NUM_THREADS=${OMP_NUM_THREADS:-<padrao>}"
    echo "compilador: $(g++ --version | head -1)"
    command -v nvcc >/dev/null && echo "nvcc:       $(nvcc --version | tail -1)"
    echo
    lscpu
    echo
    command -v nvidia-smi >/dev/null && nvidia-smi
} >"$OUT/ambiente.txt" 2>&1

if [ "${PRECOMPILADO:-0}" = 1 ]; then
    echo "== [$(hostname)] CPU: binário compilado no login (padrão das aulas)"
    [ -x ./barenn ] || { echo "FALHA: não há ./barenn compilado"; exit 1; }
else
    echo "== [$(hostname)] CPU: compilação"
    make barenn >"$OUT/compilacao_cpu.txt" 2>&1 || { echo "FALHA: não compilou"; cat "$OUT/compilacao_cpu.txt"; exit 1; }
fi
echo "== CPU: validação"
./scripts/validar.sh ./barenn | tee "$OUT/validacao_cpu.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FALHA NA VALIDAÇÃO (CPU): não meça, não crie tag"; exit 1; }
echo "== CPU: benchmark"
./scripts/bench.sh ./barenn "${REPETICOES:-10}" "$OUT/bench_cpu.csv" >"$OUT/bench_cpu.md"
tail -6 "$OUT/bench_cpu.md"
if [ "${VARREDURA_OMP:-0}" = 1 ]; then
    echo "== CPU: varredura de threads OpenMP"
    for t in 1 2 4 8 16 32 64; do
        [ "$t" -gt "$(nproc)" ] && break
        echo "### OMP_NUM_THREADS=$t" >>"$OUT/varredura_omp.md"
        OMP_NUM_THREADS=$t ./scripts/bench.sh ./barenn "${REPETICOES:-10}" | tail -6 >>"$OUT/varredura_omp.md"
    done
    cat "$OUT/varredura_omp.md"
fi
if [ "${SEM_PROFILING:-0}" != 1 ]; then
    echo "== CPU: profiling"
    OMP_NUM_THREADS=${OMP_NUM_THREADS:-} ./scripts/profile_cpu.sh ./barenn "$OUT" >/dev/null 2>&1
    head -25 "$OUT/callgrind.txt" 2>/dev/null
fi

if [ -f barenn_cuda.cu ]; then
    if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
        echo "AVISO: há barenn_cuda.cu, mas este nó não tem GPU visível"
    else
        if [ "${PRECOMPILADO:-0}" = 1 ]; then
            [ -x ./barenn_cuda ] || { echo "FALHA: não há ./barenn_cuda compilado"; exit 1; }
        else
            echo "== GPU: compilação"
            make cuda ${CUDA_ARCH:+CUDA_ARCH=$CUDA_ARCH} ${BLOCO_TREINO:+BLOCO_TREINO=$BLOCO_TREINO} \
                >"$OUT/compilacao_cuda.txt" 2>&1 || { echo "FALHA: CUDA não compilou"; cat "$OUT/compilacao_cuda.txt"; exit 1; }
        fi
        echo "== GPU: validação"
        ./scripts/validar.sh ./barenn_cuda | tee "$OUT/validacao_cuda.txt"
        [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FALHA NA VALIDAÇÃO (GPU): não meça, não crie tag"; exit 1; }
        echo "== GPU: benchmark"
        ./scripts/bench.sh ./barenn_cuda "${REPETICOES:-10}" "$OUT/bench_cuda.csv" >"$OUT/bench_cuda.md"
        tail -6 "$OUT/bench_cuda.md"
        # Varredura de block size do treino: binários barenn_cuda_b<N> compilados
        # no login pelo franky.sh. Cada um precisa passar na validação antes de medir.
        for b in ${VARREDURA_BLOCO:-}; do
            bin=./barenn_cuda_b$b
            [ -x "$bin" ] || { echo "FALHA: não há $bin compilado"; exit 1; }
            echo "== GPU: block size $b (validação + benchmark)"
            ./scripts/validar.sh "$bin" >"$OUT/validacao_cuda_b$b.txt" 2>&1
            [ $? -eq 0 ] || { cat "$OUT/validacao_cuda_b$b.txt"; echo "FALHA NA VALIDAÇÃO (GPU, bloco $b): não meça, não crie tag"; exit 1; }
            tail -1 "$OUT/validacao_cuda_b$b.txt"
            echo "### BLOCO_TREINO=$b" >>"$OUT/varredura_bloco.md"
            ./scripts/bench.sh "$bin" "${REPETICOES:-10}" "$OUT/bench_cuda_b$b.csv" | tail -6 >>"$OUT/varredura_bloco.md"
        done
        [ -f "$OUT/varredura_bloco.md" ] && cat "$OUT/varredura_bloco.md"
        if [ "${SEM_PROFILING:-0}" != 1 ] && [ -x scripts/profile_gpu.sh ]; then
            echo "== GPU: profiling (nsys/ncu)"
            ./scripts/profile_gpu.sh ./barenn_cuda "$OUT" >"$OUT/profile_gpu.log" 2>&1
            head -40 "$OUT/nsys_stats.txt" 2>/dev/null
        fi
    fi
fi
echo "OK: resultados em $OUT"
