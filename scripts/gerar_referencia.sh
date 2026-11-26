#!/usr/bin/env bash
# Regenera as saídas de referência em testes/ a partir do binário LEGADO.
# Só faz sentido com o código da tag v0.0-baseline (comportamento original):
#   git checkout v0.0-baseline && make && scripts/gerar_referencia.sh ./barenn
# Nunca gere a referência a partir de uma versão otimizada.
set -euo pipefail
BIN=${1:?binario legado}
RAIZ=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$RAIZ/testes"
for caso in tra tra2; do
    BARENN_VERIFICAR=1 "$BIN" "$RAIZ/optdigits.$caso" "$RAIZ/optdigits.tes" \
        >"$RAIZ/testes/saida.tmp" 2>"$RAIZ/testes/verif.tmp"
    grep -v -e '^Tempo' -e '^#' "$RAIZ/testes/saida.tmp" >"$RAIZ/testes/referencia_$caso.txt"
    grep -e '^perda ' -e '^camada ' "$RAIZ/testes/verif.tmp" >"$RAIZ/testes/referencia_$caso.verif"
done
rm -f "$RAIZ/testes/saida.tmp" "$RAIZ/testes/verif.tmp"
echo "referências regeneradas em $RAIZ/testes"
