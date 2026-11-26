## Resumo executivo
<!-- 2 a 3 frases: o que esta mudança entrega, o número principal (nas duas filas) e se é seguro integrar. -->

Fecha #<!-- número da Issue -->

## Resultado nas duas filas
| Fila (hardware) | Antes (s) | Depois (s) | Speedup | Validação |
|:--|--:|--:|--:|:--|
| <!-- fila A --> | | | | |
| <!-- fila B --> | | | | |

<!-- média ± desvio de 10 repetições (scripts/bench.sh); dados crus em resultados/<tag>/<fila>/ -->

## O que mudou no código e por quê
<!-- um item por commit: o que mudou e qual gargalo do diagnóstico ele ataca -->

## Riscos e corretude
- [ ] `scripts/validar.sh` aprovado nas duas filas (bit-exato ou desvio justificado)
- [ ] Compila do zero com `make` (e `make cuda`) num checkout limpo
- [ ] Nenhuma mudança de comportamento visível (saída padrão idêntica à do legado)
<!-- riscos residuais: o que pode quebrar, em que condições -->

## Como reproduzir
```bash
git checkout <tag>
make && make validar && scripts/bench.sh ./barenn 10
scripts/franky.sh    # nas duas filas, no cluster Franky
```
