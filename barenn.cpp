#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;

class DenseLayer {
    int in_features;
    int out_features;
    bool final;
    // A primeira camada não precisa devolver gradiente: ninguém o consome.
    bool propaga_gradiente;
    // Pesos em um único bloco contíguo, transposto: wt[j * out_features + i]
    // liga a entrada j ao neurônio de saída i. Para uma entrada fixa j, os pesos
    // de todos os neurônios ficam lado a lado, então os laços internos andam com
    // passo 1 e o compilador os vetoriza (SIMD) sem mudar a ordem de nenhuma
    // soma: cada neurônio continua acumulando as entradas na ordem j = 0, 1, ...
    vector<float> wt;
    vector<float> bias;
    // Dados do último forward usados para calcular os gradientes. A entrada
    // não é copiada: aponta para a amostra ou para a saída da camada anterior,
    // que permanecem inalteradas até o Backward desta camada.
    const float* last_input = nullptr;
    vector<float> last_z;
    // Buffers pré-alocados e reutilizados a cada amostra (sem malloc/free no laço).
    vector<float> out;
    vector<float> delta;
    vector<float> grad_in;

public:
    DenseLayer(int input_size, int output_size, bool is_final = false,
               bool propaga = true)
        : in_features(input_size), out_features(output_size), final(is_final),
          propaga_gradiente(propaga),
          wt(static_cast<size_t>(input_size) * output_size), bias(output_size, 0),
          last_z(output_size), out(output_size), delta(output_size),
          grad_in(input_size) {
        // A semente fixa torna a inicialização de cada camada reproduzível.
        // Sorteia na mesma ordem lógica do legado (neurônio i, entrada j) para
        // que cada peso receba exatamente o mesmo valor inicial.
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                wt[static_cast<size_t>(j) * out_features + i] = distribution(gen);
            }
        }
    }

    const vector<float>& Forward(const vector<float>& input) {
        last_input = input.data();
        const float* __restrict x = input.data();
        float* __restrict z = last_z.data();
        // Cada neurônio soma o viés aos produtos entre entradas e pesos.
        for (int i = 0; i < out_features; ++i) {
            z[i] = bias[i];
        }
        for (int j = 0; j < in_features; ++j) {
            const float xj = x[j];
            const float* __restrict w = &wt[static_cast<size_t>(j) * out_features];
            for (int i = 0; i < out_features; ++i) {
                z[i] += xj * w[i];
            }
        }
        // As camadas ocultas usam ReLU. A última devolve escores lineares.
        for (int i = 0; i < out_features; ++i) {
            out[i] = final ? z[i] : max(0.0f, z[i]);
        }
        return out;
    }

    const vector<float>& Backward(const vector<float>& grad_out,
                                  float learning_rate) {
        const float* __restrict x = last_input;
        float* __restrict d = delta.data();
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        for (int i = 0; i < out_features; ++i) {
            d[i] = final ? grad_out[i]
                         : grad_out[i] * (last_z[i] > 0.0f ? 1.0f : 0.0f);
        }

        // O gradiente da entrada usa os pesos anteriores à atualização.
        // Com o layout transposto, a linha j de wt é contígua.
        if (propaga_gradiente) {
            for (int j = 0; j < in_features; ++j) {
                const float* __restrict w = &wt[static_cast<size_t>(j) * out_features];
                float sum = 0;
                for (int i = 0; i < out_features; ++i) {
                    sum += d[i] * w[i];
                }
                grad_in[j] = sum;
            }
        }

        // SGD por amostra. O gradiente do peso é delta vezes a entrada.
        // O gradiente do viés é o próprio delta. Mesma expressão do legado,
        // lr * (delta * x), para preservar o arredondamento.
        for (int j = 0; j < in_features; ++j) {
            const float xj = x[j];
            float* __restrict w = &wt[static_cast<size_t>(j) * out_features];
            for (int i = 0; i < out_features; ++i) {
                w[i] -= learning_rate * (d[i] * xj);
            }
        }
        for (int i = 0; i < out_features; ++i) {
            bias[i] -= learning_rate * d[i];
        }
        return grad_in;
    }

    // Resumo dos parâmetros na ordem lógica [saída][entrada], seguido do viés.
    // Usado apenas pelo modo de verificação (BARENN_VERIFICAR=1): o hash
    // FNV-1a cobre os bits exatos de cada float, a soma dá a ordem de grandeza.
    void Resumo(uint64_t& hash, double& soma) const {
        auto mistura = [&](float valor) {
            uint32_t bits;
            memcpy(&bits, &valor, sizeof bits);
            hash = (hash ^ bits) * 1099511628211ULL;
            soma += valor;
        };
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                mistura(wt[static_cast<size_t>(j) * out_features + i]);
            }
        }
        for (int i = 0; i < out_features; ++i) {
            mistura(bias[i]);
        }
    }
};

struct Amostra {
    vector<float> imagem;
    int rotulo;
};

vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    vector<Amostra> dados;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        Amostra amostra{};
        // Cada atributo deve ser seguido por vírgula, inclusive o último.
        for (int i = 0; i < entradas; ++i) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            // Converte os valores de [0, 16] para [0, 1].
            amostra.imagem.push_back(valor / 16.0f);
        }
        // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
        if (amostra.imagem.size() != entradas ||
            !(campos >> amostra.rotulo) ||
            amostra.rotulo < 0 || amostra.rotulo >= classes ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.push_back(move(amostra));
    }
    if (arquivo.bad() || dados.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
float perda(const vector<float>& escores, int rotulo,
            vector<float>* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            (*gradiente)[i] = erro;
        }
    }
    return total;
}

void avaliar(DenseLayer& l1, DenseLayer& l2, DenseLayer& l3,
             const vector<Amostra>& teste) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    // Só fazemos forward aqui. Os pesos ficam fixos.
    for (size_t i = 0; i < teste.size(); ++i) {
        const auto& amostra = teste[i];
        const auto& escores = l3.Forward(l2.Forward(l1.Forward(amostra.imagem)));
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(
            max_element(escores.begin(), escores.end()) - escores.begin());
        bool acertou = previsto == amostra.rotulo;
        acertos += acertou;
        ++confusoes[amostra.rotulo][previsto];
        perda_total += perda(escores, amostra.rotulo);

        // Guarda até três acertos e três erros para ilustrar a saída.
        // A acurácia usa todas as imagens, inclusive as não exibidas.
        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            int segundo = previsto == 0 ? 1 : 0;
            for (int classe = 0; classe < classes; ++classe) {
                if (classe != previsto && escores[classe] > escores[segundo]) {
                    segundo = classe;
                }
            }
            exemplos.push_back({i + 1, amostra.rotulo, previsto, segundo,
                                escores[previsto], escores[segundo]});
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size() << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}

// Modo de verificação: com BARENN_VERIFICAR definido, imprime no stderr a perda
// exata de todas as épocas e o resumo bit a bit dos pesos finais. A saída padrão
// não muda. Os scripts de validação comparam esse relatório com o do legado.
template <size_t N>
void relatorio_verificacao(const array<double, N>& perdas,
                           const DenseLayer& l1, const DenseLayer& l2,
                           const DenseLayer& l3) {
    if (!getenv("BARENN_VERIFICAR")) {
        return;
    }
    cerr << hexfloat;
    for (size_t epoca = 0; epoca < N; ++epoca) {
        cerr << "perda " << epoca + 1 << " " << perdas[epoca] << "\n";
    }
    const DenseLayer* camadas[] = {&l1, &l2, &l3};
    for (int c = 0; c < 3; ++c) {
        uint64_t hash = 1469598103934665603ULL;
        double soma = 0;
        camadas[c]->Resumo(hash, soma);
        cerr << "camada " << c + 1 << " hash " << hex << hash << dec
             << " soma " << soma << "\n";
    }
    cerr << defaultfloat;
}

} // namespace Barenn

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        const auto inicio_leitura = chrono::steady_clock::now();
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        const chrono::duration<double> tempo_leitura =
            chrono::steady_clock::now() - inicio_leitura;
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        // A primeira camada não propaga gradiente (a entrada é o dado bruto).
        Barenn::DenseLayer l1(Barenn::entradas, 32, false, false);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);
        vector<float> gradiente(Barenn::classes);
        array<double, epocas> perdas{};

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (const auto& amostra : treino) {
                const auto& escores =
                    l3.Forward(l2.Forward(l1.Forward(amostra.imagem)));
                perda_total += Barenn::perda(escores, amostra.rotulo, &gradiente);

                // Propaga o gradiente da última camada até a primeira.
                // Cada Backward também atualiza os parâmetros da sua camada.
                const auto& grad3 = l3.Backward(gradiente, taxa);
                const auto& grad2 = l2.Backward(grad3, taxa);
                l1.Backward(grad2, taxa);
            }
            perdas[epoca] = perda_total / treino.size();
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;
        Barenn::relatorio_verificacao(perdas, l1, l2, l3);

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
        const auto inicio_avaliacao = chrono::steady_clock::now();
        Barenn::avaliar(l1, l2, l3, teste);
        const chrono::duration<double> tempo_avaliacao =
            chrono::steady_clock::now() - inicio_avaliacao;
        // Tempos das demais fases, para medir a fração serial (Amdahl).
        cout << "Tempo de leitura " << tempo_leitura.count() << " s\n"
             << "Tempo de avaliacao " << tempo_avaliacao.count() << " s\n";
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
