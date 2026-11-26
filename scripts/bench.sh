#!/usr/bin/env bash
# Benchmark com repetições e estatística.
#
# Uso: scripts/bench.sh [binario] [repeticoes] [arquivo.csv]
#      (padrão: ./barenn, 10 repetições, sem CSV)
#
# Faz 1 execução de aquecimento (descartada) e N execuções medidas. Para cada uma
# registra o "Tempo de treino" impresso pelo programa e o tempo total do processo
# (leitura + treino + avaliação). Imprime ambiente, média, desvio padrão amostral,
# mínimo, mediana e máximo, em formato de tabela Markdown pronta para a Issue.
set -u
export LC_ALL=C  # ponto decimal em awk/bc independentemente do locale

BIN=${1:-./barenn}
N=${2:-10}
CSV=${3:-}
RAIZ=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "### Ambiente"
echo '```'
echo "data:      $(date -Iseconds)"
echo "host:      $(hostname)"
echo "versao:    ${VERSAO:-$(git -C "$RAIZ" describe --tags --always --dirty 2>/dev/null || echo "?")}"
echo "binario:   $BIN"
echo "cpu:       $(lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1) ($(nproc) threads lógicas)"
echo "kernel:    $(uname -sr)"
echo "compilador: $(g++ --version | head -1)"
command -v nvcc >/dev/null && echo "nvcc:      $(nvcc --version | tail -1)"
command -v nvidia-smi >/dev/null && echo "gpu:       $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-<nao definido>}"
echo "entrada:   optdigits.tra (treino) / optdigits.tes (teste), $N repetições + 1 aquecimento"
echo '```'

cd "$RAIZ" || exit 1
"$BIN" >/dev/null 2>&1 || { echo "[FALHA] o binário terminou com erro"; exit 1; }

[ -n "$CSV" ] && echo "repeticao,treino_s,total_s,leitura_s,avaliacao_s" >"$CSV"
for i in $(seq 1 "$N"); do
    t0=$(date +%s.%N)
    "$BIN" >"$TMP/out" 2>/dev/null || { echo "[FALHA] execução $i terminou com erro"; exit 1; }
    t1=$(date +%s.%N)
    treino=$(sed -n 's/^Tempo de treino \([0-9.]*\) s.*/\1/p' "$TMP/out")
    total=$(echo "$t1 - $t0" | bc -l)
    leitura=$(sed -n 's/^Tempo de leitura \([0-9.]*\) s.*/\1/p' "$TMP/out")
    avaliacao=$(sed -n 's/^Tempo de avaliacao \([0-9.]*\) s.*/\1/p' "$TMP/out")
    echo "$treino $total ${leitura:-0} ${avaliacao:-0}" >>"$TMP/medidas"
    [ -n "$CSV" ] && echo "$i,$treino,$total,$leitura,$avaliacao" >>"$CSV"
done

estat() {  # coluna -> "media desvio min mediana max"
    awk -v c="$1" '{print $c}' "$TMP/medidas" | sort -g | awk '
        {v[NR]=$1; s+=$1}
        END {
            m=s/NR; for(i=1;i<=NR;i++) q+=(v[i]-m)^2
            sd=(NR>1)?sqrt(q/(NR-1)):0
            med=(NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2
            printf "%.4f | %.4f | %.4f | %.4f | %.4f", m, sd, v[1], med, v[NR]
        }'
}

echo
echo "### Resultado ($N repetições)"
echo "| Métrica | Média (s) | Desvio (s) | Mín (s) | Mediana (s) | Máx (s) |"
echo "|:--|--:|--:|--:|--:|--:|"
echo "| Tempo de treino (cronômetro interno) | $(estat 1) |"
echo "| Tempo de leitura dos arquivos | $(estat 3) |"
echo "| Tempo de avaliação (teste) | $(estat 4) |"
echo "| Tempo total do processo | $(estat 2) |"
