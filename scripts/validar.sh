#!/usr/bin/env bash
# Valida a corretude de um binário do Barenn contra a saída do código legado.
#
# Uso: scripts/validar.sh [binario]          (padrão: ./barenn)
#
# Para cada conjunto de treino (optdigits.tra e optdigits.tra2) o binário roda
# com BARENN_VERIFICAR=1 e é comparado com testes/referencia_<caso>.*:
#   - stdout (sem as linhas de tempo) deve ser IDÊNTICO ao do legado;
#   - o relatório de verificação (stderr: perda exata de cada época em hexfloat
#     e hash FNV-1a dos bits de todos os pesos) indica se o resultado é
#     BIT-EXATO ou apenas equivalente dentro de tolerância.
# Também confere que entradas inválidas continuam sendo rejeitadas.
# Sai com código != 0 se qualquer verificação falhar.
set -u
export LC_ALL=C  # ponto decimal em awk/bc independentemente do locale

BIN=${1:-./barenn}
RAIZ=$(cd "$(dirname "$0")/.." && pwd)
REF="$RAIZ/testes"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

falhas=0

if [ ! -x "$BIN" ]; then
    echo "[FALHA] binário '$BIN' não existe ou não é executável"
    exit 1
fi

for caso in tra tra2; do
    BARENN_VERIFICAR=1 "$BIN" "$RAIZ/optdigits.$caso" "$RAIZ/optdigits.tes" \
        >"$TMP/out" 2>"$TMP/err"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FALHA] $caso: o binário terminou com código $rc"
        cat "$TMP/err"
        falhas=$((falhas + 1))
        continue
    fi
    # Linhas de tempo e linhas informativas (começando com '#') variam a cada execução.
    grep -v -e '^Tempo' -e '^#' "$TMP/out" >"$TMP/out.limpo"
    grep -e '^perda ' -e '^camada ' "$TMP/err" >"$TMP/verif"

    if ! diff -u "$REF/referencia_$caso.txt" "$TMP/out.limpo" >"$TMP/diff"; then
        echo "[FALHA] $caso: saída diferente do legado"
        cat "$TMP/diff"
        falhas=$((falhas + 1))
        continue
    fi

    if diff -q "$REF/referencia_$caso.verif" "$TMP/verif" >/dev/null; then
        echo "[OK] $caso: BIT-EXATO (saída, perdas de todas as épocas e bits de todos os pesos idênticos ao legado)"
    else
        # Saída visível idêntica, mas os bits divergem: mostra a maior diferença.
        python3 - "$REF/referencia_$caso.verif" "$TMP/verif" <<'EOF'
import sys
def ler(caminho):
    perdas, somas = {}, {}
    for linha in open(caminho):
        p = linha.split()
        if p[0] == "perda":
            perdas[int(p[1])] = float.fromhex(p[2])
        elif p[0] == "camada":
            somas[int(p[1])] = float.fromhex(p[5])
    return perdas, somas
pr, sr = ler(sys.argv[1])
pn, sn = ler(sys.argv[2])
dp = max(abs(pr[k] - pn.get(k, float("inf"))) / abs(pr[k]) for k in pr)
ds = max(abs(sr[k] - sn.get(k, float("inf"))) for k in sr)
print(f"[OK*] saída idêntica, mas NÃO bit-exato: maior desvio relativo da perda "
      f"por época = {dp:.3e}; maior desvio na soma dos pesos = {ds:.3e}. "
      f"Justifique na Issue (ordem de soma / FMA).")
EOF
    fi
done

# Entradas inválidas continuam rejeitadas com código 1.
"$BIN" "$RAIZ/optdigits.tes" "$RAIZ/optdigits.tes" >/dev/null 2>&1
[ $? -eq 1 ] && echo "[OK] rejeita treino == teste" || { echo "[FALHA] aceitou treino == teste"; falhas=$((falhas + 1)); }
"$BIN" "$RAIZ/optdigits.names" "$RAIZ/optdigits.tes" >/dev/null 2>&1
[ $? -eq 1 ] && echo "[OK] rejeita arquivo malformado" || { echo "[FALHA] aceitou arquivo malformado"; falhas=$((falhas + 1)); }
"$BIN" so_um_argumento >/dev/null 2>&1
[ $? -eq 1 ] && echo "[OK] rejeita argumentos inválidos" || { echo "[FALHA] aceitou argumentos inválidos"; falhas=$((falhas + 1)); }

if [ $falhas -ne 0 ]; then
    echo "VALIDACAO REPROVADA ($falhas falha(s)) - NAO crie tag nem faça merge."
    exit 1
fi
echo "VALIDACAO APROVADA"
