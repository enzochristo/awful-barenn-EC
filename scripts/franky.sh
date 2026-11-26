#!/usr/bin/env bash
# Roda a versão atual (commit em que você está) nas DUAS filas da Franky,
# um job SLURM por fila, e espera os dois terminarem.
#
# Uso (no nó de login, na raiz do repositório):
#   FILAS="fila1 fila2" VERSAO=v0.0-baseline scripts/franky.sh
#   VARREDURA_OMP=1 scripts/franky.sh       # + varredura de threads OpenMP
# Resultados: resultados/<versao>/<fila>/  (ambiente, validação, bench, profiling)
# Logs do SLURM: outputs/<versao>/<fila>/slurm-<job>.out
#
# Os jobs pedem o nó inteiro (--exclusive) para que nenhum outro processo
# interfira nas medições. Use EXCLUSIVO=0 se a fila não permitir.
set -u
FILAS=${FILAS:-"FILA_A FILA_B"}      # <- trocar pelos nomes reais das duas filas
TEMPO=${TEMPO:-00:19:00}     # abaixo do limite de 20 min do monstrao (as versões GPU ingênuas são lentas)
RAIZ=$(cd "$(dirname "$0")/.." && pwd)
cd "$RAIZ" || exit 1
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "ERRO: há alterações não commitadas; meça só código commitado."; exit 1
fi
# Nome da pasta de resultados: a tag que esta versão vai receber (ex.: v0.0-baseline).
VERSAO=${VERSAO:-$(git describe --tags --always)}
echo "versão: $VERSAO (commit $(git rev-parse --short HEAD))"

# Padrão das aulas: compila no LOGIN com g++ (e nvcc) ANTES de submeter; os jobs só
# executam o binário. (Dentro do nó o g++ do OpenHPC não achava features.h.)
if [ -f barenn_cuda.cu ] && ! command -v nvcc >/dev/null 2>&1; then
    [ -f /etc/profile.d/lmod.sh ] && { set +u; . /etc/profile.d/lmod.sh; set -u; }
    command -v module >/dev/null 2>&1 && { set +u; module load cuda/12.8.1; set -u; }
fi
for fila in $FILAS; do mkdir -p "resultados/$VERSAO/$fila"; done
echo ">> compilando no login: make barenn"
make barenn >"resultados/$VERSAO/compilacao_cpu.txt" 2>&1 || { cat "resultados/$VERSAO/compilacao_cpu.txt"; echo "ERRO: não compilou"; exit 1; }
if [ -f barenn_cuda.cu ]; then
    echo ">> compilando no login: make cuda"
    make cuda ${CUDA_ARCH:+CUDA_ARCH=$CUDA_ARCH} ${BLOCO_TREINO:+BLOCO_TREINO=$BLOCO_TREINO} \
        >"resultados/$VERSAO/compilacao_cuda.txt" 2>&1 || { cat "resultados/$VERSAO/compilacao_cuda.txt"; echo "ERRO: CUDA não compilou"; exit 1; }
fi
for fila in $FILAS; do cp resultados/$VERSAO/compilacao_*.txt "resultados/$VERSAO/$fila/"; done
rm -f resultados/$VERSAO/compilacao_*.txt

extra=()
[ "${EXCLUSIVO:-1}" = 1 ] && extra+=(--exclusive)
[ -f barenn_cuda.cu ] && extra+=(--gres="${GRES:-gpu:1}")

pids=()
for fila in $FILAS; do
    out="resultados/$VERSAO/$fila"
    log="outputs/$VERSAO/$fila"
    mkdir -p "$out" "$log"
    echo ">> submetendo $VERSAO na fila $fila"
    sbatch --wait --partition="$fila" --job-name="barenn-$fila" --time="$TEMPO" \
        "${extra[@]}" --output="$log/slurm-%j.out" \
        --export=ALL,VERSAO="$VERSAO",PRECOMPILADO=1 \
        --wrap "scripts/coleta.sh $out" &
    pids+=($!)
done
falhas=0
for p in "${pids[@]}"; do wait "$p" || falhas=$((falhas + 1)); done

echo
for fila in $FILAS; do
    out="resultados/$VERSAO/$fila"
    echo "================ $fila ($(sed -n 's/^no: *//p' "$out/ambiente.txt" 2>/dev/null))"
    grep -h -e 'VALIDACAO' "$out"/validacao_*.txt 2>/dev/null
    grep -h -e '^| Tempo' "$out"/bench_*.md 2>/dev/null
done
[ $falhas -eq 0 ] || { echo "ATENÇÃO: $falhas job(s) falharam; veja outputs/$VERSAO/*/slurm-*.out"; exit 1; }
echo "Tudo certo. Envie as pastas resultados/$VERSAO e outputs/$VERSAO para o repositório."
