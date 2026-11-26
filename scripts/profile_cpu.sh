#!/usr/bin/env bash
# Profiling de CPU: callgrind (instruções por função) e perf (contadores de hardware).
#
# Uso: scripts/profile_cpu.sh [binario] [pasta_saida]
#      (padrão: ./barenn, resultados/<versao>)
#
# Gera na pasta de saída:
#   callgrind.txt   top de funções por instruções executadas (Ir)
#   perf_stat.txt   ciclos, IPC, cache-misses, L1, branch-misses (5 repetições)
#   perf_report.txt top de funções por amostragem de ciclos
# O perf exige: sudo sysctl kernel.perf_event_paranoid=1
set -u
export LC_ALL=C  # ponto decimal em awk/bc independentemente do locale

BIN=$(realpath "${1:-./barenn}")
RAIZ=$(cd "$(dirname "$0")/.." && pwd)
VERSAO=$(git -C "$RAIZ" describe --tags --always --dirty 2>/dev/null || echo atual)
OUT=$(realpath -m "${2:-$RAIZ/resultados/$VERSAO}")
mkdir -p "$OUT"
cd "$RAIZ" || exit 1

if command -v valgrind >/dev/null; then
    echo ">> callgrind (pode levar ~1 min)"
    valgrind --tool=callgrind --callgrind-out-file="$OUT/callgrind.out" \
        "$BIN" >/dev/null 2>"$OUT/callgrind.log"
    callgrind_annotate --inclusive=no "$OUT/callgrind.out" 2>/dev/null | head -40 >"$OUT/callgrind.txt"
    rm -f "$OUT/callgrind.out"
    cat "$OUT/callgrind.txt"
else
    echo "valgrind não encontrado"
fi

if command -v perf >/dev/null; then
    echo ">> perf stat"
    perf stat -r 5 -e task-clock,cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,branches,branch-misses \
        "$BIN" >/dev/null 2>"$OUT/perf_stat.txt"
    cat "$OUT/perf_stat.txt"
    echo ">> perf record"
    perf record -q -g -o "$OUT/perf.data" "$BIN" >/dev/null 2>&1 &&
        perf report -i "$OUT/perf.data" --no-children --stdio 2>/dev/null |
        grep -E '^ +[0-9]' | head -20 >"$OUT/perf_report.txt"
    rm -f "$OUT/perf.data"
    cat "$OUT/perf_report.txt"
else
    echo "perf não encontrado"
fi
echo "Resultados em $OUT"
