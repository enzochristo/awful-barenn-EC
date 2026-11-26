#!/usr/bin/env bash
# Profiling de GPU com Nsight Systems (linha do tempo) e Nsight Compute (kernel).
#
# Uso: scripts/profile_gpu.sh [binario] [pasta_saida]
#      (padrão: ./barenn_cuda, resultados/<versao>)
# Variáveis opcionais:
#   NCU_TREINO   arquivo de treino usado no ncu (padrão optdigits.tra2: poucas
#                amostras, para o ncu não instrumentar milhões de lançamentos;
#                as métricas por lançamento são as mesmas). Use optdigits.tra
#                para o kernel persistente (um único lançamento).
#   NCU_KERNELS  quantos lançamentos o ncu analisa (padrão 12)
#   NSYS_AMOSTRAS quantas amostras de treino o nsys percorre (padrão 200). Na versão
#                com um lançamento por fase e por amostra, o treino completo geraria
#                milhões de eventos; 200 amostras × 51 épocas já mostram o padrão
#                por amostra, que se repete igual no treino inteiro.
#
# Gera na pasta de saída:
#   nsys_stats.txt    tempo por kernel, por transferência (H2D/D2H) e por API
#   nsys.nsys-rep     linha do tempo (abra no Nsight Systems para print)
#   ncu_detalhes.txt  SOL, ocupação teórica x real, memória, stalls, por kernel
#   ncu.ncu-rep       relatório completo (abra no Nsight Compute para print)
set -u
export LC_ALL=C

BIN=$(realpath "${1:-./barenn_cuda}")
RAIZ=$(cd "$(dirname "$0")/.." && pwd)
VERSAO=$(git -C "$RAIZ" describe --tags --always --dirty 2>/dev/null || echo atual)
OUT=$(realpath -m "${2:-$RAIZ/resultados/$VERSAO}")
mkdir -p "$OUT"
cd "$RAIZ" || exit 1

nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,clocks.max.sm,pcie.link.gen.max,pcie.link.width.max \
    --format=csv >"$OUT/gpu_info.txt" 2>&1
cat "$OUT/gpu_info.txt"

if command -v nsys >/dev/null; then
    TMPD=$(mktemp -d); trap 'rm -rf "$TMPD"' EXIT
    head -n "${NSYS_AMOSTRAS:-200}" optdigits.tra >"$TMPD/treino_nsys.txt"
    echo ">> nsys (treino com as primeiras ${NSYS_AMOSTRAS:-200} amostras)"
    nsys profile --trace=cuda,osrt --force-overwrite=true -o "$OUT/nsys" \
        "$BIN" "$TMPD/treino_nsys.txt" optdigits.tes >"$OUT/nsys_saida.txt" 2>&1
    # Nomes dos relatórios mudaram entre versões do nsys: tenta os dois.
    nsys stats --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
        --report cuda_gpu_mem_size_sum --report cuda_api_sum \
        "$OUT/nsys.nsys-rep" >"$OUT/nsys_stats.txt" 2>&1 ||
    nsys stats --report gpukernsum --report gpumemtimesum \
        --report gpumemsizesum --report cudaapisum \
        "$OUT/nsys.nsys-rep" >"$OUT/nsys_stats.txt" 2>&1
    rm -f "$OUT"/nsys.sqlite
    cat "$OUT/nsys_stats.txt"
else
    echo "nsys não encontrado (module load cuda / nsight-systems?)"
fi

if command -v ncu >/dev/null; then
    echo ">> ncu (${NCU_KERNELS:-12} lançamentos, treino ${NCU_TREINO:-optdigits.tra2})"
    ncu --set full -k regex:"k_" -c "${NCU_KERNELS:-12}" --force-overwrite \
        -o "$OUT/ncu" "$BIN" "${NCU_TREINO:-optdigits.tra2}" optdigits.tes \
        >"$OUT/ncu_saida.txt" 2>&1
    ncu --import "$OUT/ncu.ncu-rep" --page details >"$OUT/ncu_detalhes.txt" 2>&1
    # Resumo das métricas mais citadas nas Issues.
    grep -E '^  [a-zA-Z_:]+.*\(|Duration|Theoretical Occupancy|Achieved Occupancy|Compute \(SM\) Throughput|Memory Throughput|DRAM Throughput|Registers Per Thread|Shared Memory Configuration Size|Block Size|Grid Size|Waves Per SM|Bank' \
        "$OUT/ncu_detalhes.txt" | head -150 >"$OUT/ncu_resumo.txt"
    cat "$OUT/ncu_resumo.txt"
else
    echo "ncu não encontrado (module load cuda / nsight-compute?)"
fi
echo "Resultados em $OUT"
