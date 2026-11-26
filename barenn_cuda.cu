// Barenn em CUDA - v1.1: dados residentes e um kernel fundido por amostra.
//
// Mesma rede (64 -> 32 -> 16 -> 10), mesma inicialização, mesmo SGD por amostra
// e mesma ordem de cada soma da versão CPU. Compilado com --fmad=false para que
// a*b+c não vire FMA: assim o resultado é bit a bit igual ao do legado.
//
// Em relação ao porting inicial (v1.0):
//   - o conjunto de treino inteiro é copiado para a GPU uma única vez, e as
//     perdas de todas as amostras ficam na GPU até o fim do treino: nenhuma
//     transferência PCIe dentro do laço;
//   - forward, perda e backward de uma amostra rodam em UM kernel (um bloco,
//     fases separadas por __syncthreads), em vez de ~12 lançamentos;
//   - a avaliação calcula todas as imagens de teste em paralelo (um thread
//     por imagem) com uma única cópia de ida e uma de volta.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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

// Aborta com mensagem clara se qualquer chamada da API CUDA falhar.
#define CUDA_CHECK(chamada)                                                    \
    do {                                                                       \
        cudaError_t erro_cuda = (chamada);                                     \
        if (erro_cuda != cudaSuccess) {                                        \
            throw runtime_error(string("CUDA: ") + cudaGetErrorString(erro_cuda) + \
                                " em " #chamada " (" __FILE__ ":" +            \
                                to_string(__LINE__) + ")");                    \
        }                                                                      \
    } while (0)

namespace Barenn {

constexpr int entradas = 64;
constexpr int classes = 10;
// Threads do bloco que processa uma amostra. As fases com mais trabalho são as
// atualizações de pesos (até 64 x 32 = 2048 por camada).
#ifndef BLOCO_TREINO
#define BLOCO_TREINO 256
#endif
constexpr int bloco_inferencia = 128;

// ---------------------------------------------------------------------------
// Camada: ponteiros para parâmetros e buffers no device
// ---------------------------------------------------------------------------
struct Camada {
    float* wt;       // pesos transpostos [entrada][saída]
    float* b;        // viés
    float* z;        // soma antes da ativação
    float* out;      // saída após ativação
    float* delta;    // gradiente em relação a z
    float* grad_in;  // gradiente em relação à entrada
    int in;
    int nout;
    bool final;
};

// Cada fase distribui os neurônios/pesos entre os threads do bloco. A ordem
// de cada soma é a mesma da CPU (um único thread acumula cada valor).
__device__ void fase_forward(const Camada& c, const float* x) {
    for (int i = threadIdx.x; i < c.nout; i += blockDim.x) {
        float sum = c.b[i];
        for (int j = 0; j < c.in; ++j) {
            sum += x[j] * c.wt[j * c.nout + i];
        }
        c.z[i] = sum;
        c.out[i] = c.final ? sum : ((0.0f < sum) ? sum : 0.0f);
    }
}

__device__ void fase_delta(const Camada& c, const float* grad_out) {
    for (int i = threadIdx.x; i < c.nout; i += blockDim.x) {
        c.delta[i] = c.final ? grad_out[i]
                             : grad_out[i] * (c.z[i] > 0.0f ? 1.0f : 0.0f);
    }
}

__device__ void fase_grad_entrada(const Camada& c) {
    for (int j = threadIdx.x; j < c.in; j += blockDim.x) {
        float sum = 0;
        for (int i = 0; i < c.nout; ++i) {
            sum += c.delta[i] * c.wt[j * c.nout + i];
        }
        c.grad_in[j] = sum;
    }
}

__device__ void fase_atualiza(const Camada& c, const float* x, float lr) {
    const int total = c.in * c.nout;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int j = idx / c.nout;
        int i = idx - j * c.nout;
        c.wt[idx] -= lr * (c.delta[i] * x[j]);
    }
    for (int i = threadIdx.x; i < c.nout; i += blockDim.x) {
        c.b[i] -= lr * c.delta[i];
    }
}

// Forward, perda e backward completos de UMA amostra, em um único bloco.
// As fases independentes são executadas juntas para reduzir barreiras:
// a atualização da camada k não toca nos dados que o delta da camada k-1 lê.
__device__ void passo_sgd(const Camada& l1, const Camada& l2, const Camada& l3,
                          const float* x, int rotulo, float* grad,
                          float* perda_saida, float lr) {
    fase_forward(l1, x);
    __syncthreads();
    fase_forward(l2, l1.out);
    __syncthreads();
    fase_forward(l3, l2.out);
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0;
        for (int i = 0; i < classes; ++i) {
            float erro = l3.out[i] - (i == rotulo ? 1.0f : 0.0f);
            total += 0.5f * erro * erro;
            grad[i] = erro;
        }
        *perda_saida = total;
    }
    __syncthreads();
    fase_delta(l3, grad);
    __syncthreads();
    fase_grad_entrada(l3);           // usa os pesos antigos de l3
    __syncthreads();
    fase_atualiza(l3, l2.out, lr);
    fase_delta(l2, l3.grad_in);
    __syncthreads();
    fase_grad_entrada(l2);           // usa os pesos antigos de l2
    __syncthreads();
    fase_atualiza(l2, l1.out, lr);
    fase_delta(l1, l2.grad_in);
    __syncthreads();
    fase_atualiza(l1, x, lr);        // l1 não propaga gradiente
}

// Um lançamento por amostra; imagens, rótulos e perdas ficam na GPU.
__global__ void k_amostra(Camada l1, Camada l2, Camada l3, const float* imagens,
                          const int* rotulos, int indice, float* grad,
                          float* perdas, float lr) {
    passo_sgd(l1, l2, l3, imagens + static_cast<size_t>(indice) * entradas,
              rotulos[indice], grad, perdas + indice, lr);
}

// Inferência: um thread por imagem de teste, ativações em registradores.
__global__ void k_inferencia(Camada l1, Camada l2, Camada l3,
                             const float* imagens, int n, float* escores) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const float* x = imagens + static_cast<size_t>(k) * entradas;
    float h1[32];
    float h2[16];
    for (int i = 0; i < 32; ++i) {
        float sum = l1.b[i];
        for (int j = 0; j < entradas; ++j) sum += x[j] * l1.wt[j * 32 + i];
        h1[i] = (0.0f < sum) ? sum : 0.0f;
    }
    for (int i = 0; i < 16; ++i) {
        float sum = l2.b[i];
        for (int j = 0; j < 32; ++j) sum += h1[j] * l2.wt[j * 16 + i];
        h2[i] = (0.0f < sum) ? sum : 0.0f;
    }
    for (int i = 0; i < classes; ++i) {
        float sum = l3.b[i];
        for (int j = 0; j < 16; ++j) sum += h2[j] * l3.wt[j * classes + i];
        escores[static_cast<size_t>(k) * classes + i] = sum;
    }
}

// ---------------------------------------------------------------------------
// Alocação e inicialização das camadas (host)
// ---------------------------------------------------------------------------
Camada criar_camada(int in, int nout, bool final) {
    // Mesma inicialização da CPU: semente 1337, ordem lógica (i, j).
    vector<float> wt(static_cast<size_t>(in) * nout);
    mt19937 gen(1337);
    uniform_real_distribution<float> distribution(-0.05f, 0.05f);
    for (int i = 0; i < nout; ++i) {
        for (int j = 0; j < in; ++j) {
            wt[static_cast<size_t>(j) * nout + i] = distribution(gen);
        }
    }
    vector<float> b(nout, 0.0f);
    Camada c{};
    c.in = in;
    c.nout = nout;
    c.final = final;
    CUDA_CHECK(cudaMalloc(&c.wt, wt.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.b, nout * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.z, nout * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.out, nout * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.delta, nout * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.grad_in, in * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(c.wt, wt.data(), wt.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c.b, b.data(), b.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    return c;
}

void liberar_camada(Camada& c) {
    cudaFree(c.wt);
    cudaFree(c.b);
    cudaFree(c.z);
    cudaFree(c.out);
    cudaFree(c.delta);
    cudaFree(c.grad_in);
    c = Camada{};
}

// Resumo dos parâmetros (ordem lógica [saída][entrada], depois o viés),
// idêntico ao da versão CPU, para o modo de verificação.
void resumo_camada(const Camada& c, uint64_t& hash, double& soma) {
    vector<float> wt(static_cast<size_t>(c.in) * c.nout);
    vector<float> b(c.nout);
    CUDA_CHECK(cudaMemcpy(wt.data(), c.wt, wt.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), c.b, b.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    auto mistura = [&](float valor) {
        uint32_t bits;
        memcpy(&bits, &valor, sizeof bits);
        hash = (hash ^ bits) * 1099511628211ULL;
        soma += valor;
    };
    for (int i = 0; i < c.nout; ++i) {
        for (int j = 0; j < c.in; ++j) {
            mistura(wt[static_cast<size_t>(j) * c.nout + i]);
        }
    }
    for (int i = 0; i < c.nout; ++i) {
        mistura(b[i]);
    }
}

// ---------------------------------------------------------------------------
// Código de host: leitura, perda e relatório (idênticos à versão CPU)
// ---------------------------------------------------------------------------
struct Amostra {
    vector<float> imagem;
    int rotulo;
};

bool converter_linha(const string& linha, Amostra& amostra) {
    istringstream campos(linha);
    amostra.imagem.clear();
    amostra.imagem.reserve(entradas);
    for (int i = 0; i < entradas; ++i) {
        float valor;
        char separador;
        if (!(campos >> valor >> separador) || separador != ',' ||
            !isfinite(valor) || valor < 0 || valor > 16) {
            break;
        }
        amostra.imagem.push_back(valor / 16.0f);
    }
    return amostra.imagem.size() == entradas &&
           (campos >> amostra.rotulo) &&
           amostra.rotulo >= 0 && amostra.rotulo < classes &&
           (campos >> ws).eof();
}

vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }
    vector<string> linhas;
    string linha;
    while (getline(arquivo, linha)) {
        linhas.push_back(move(linha));
    }
    const long n = static_cast<long>(linhas.size());
    vector<Amostra> dados(n);
    vector<char> valida(n);
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) {
        valida[i] = converter_linha(linhas[i], dados[i]);
    }
    for (long i = 0; i < n; ++i) {
        if (!valida[i]) {
            throw runtime_error(caminho + ":" + to_string(i + 1) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
    }
    if (arquivo.bad() || dados.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

float perda(const float* escores, int rotulo) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
    }
    return total;
}

// Agrega acurácia, perda, exemplos e confusões a partir dos escores já
// calculados (na GPU), na mesma ordem e com a mesma saída da versão CPU.
void relatorio_avaliacao(const vector<array<float, classes>>& todos_escores,
                         const vector<Amostra>& teste) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    for (size_t i = 0; i < teste.size(); ++i) {
        const auto& amostra = teste[i];
        const auto& escores = todos_escores[i];
        int previsto = static_cast<int>(
            max_element(escores.begin(), escores.end()) - escores.begin());
        bool acertou = previsto == amostra.rotulo;
        acertos += acertou;
        ++confusoes[amostra.rotulo][previsto];
        perda_total += perda(escores.data(), amostra.rotulo);

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

template <size_t N>
void relatorio_verificacao(const array<double, N>& perdas, const Camada& l1,
                           const Camada& l2, const Camada& l3) {
    if (!getenv("BARENN_VERIFICAR")) {
        return;
    }
    cerr << hexfloat;
    for (size_t epoca = 0; epoca < N; ++epoca) {
        cerr << "perda " << epoca + 1 << " " << perdas[epoca] << "\n";
    }
    const Camada* camadas[] = {&l1, &l2, &l3};
    for (int c = 0; c < 3; ++c) {
        uint64_t hash = 1469598103934665603ULL;
        double soma = 0;
        resumo_camada(*camadas[c], hash, soma);
        cerr << "camada " << c + 1 << " hash " << hex << hash << dec
             << " soma " << soma << "\n";
    }
    cerr << defaultfloat;
}

// Junta as imagens em um único bloco contíguo [amostra][atributo] e os rótulos
// em um vetor, prontos para uma única cópia Host->Device.
void empacotar(const vector<Amostra>& dados, vector<float>& imagens,
               vector<int>& rotulos) {
    imagens.resize(dados.size() * entradas);
    rotulos.resize(dados.size());
    for (size_t k = 0; k < dados.size(); ++k) {
        copy(dados[k].imagem.begin(), dados[k].imagem.end(),
             imagens.begin() + k * entradas);
        rotulos[k] = dados[k].rotulo;
    }
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

        // Cria o contexto CUDA fora das medições.
        const auto inicio_gpu = chrono::steady_clock::now();
        CUDA_CHECK(cudaFree(nullptr));
        cudaDeviceProp prop{};
        int dispositivo = 0;
        CUDA_CHECK(cudaGetDevice(&dispositivo));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dispositivo));
        const chrono::duration<double> tempo_contexto =
            chrono::steady_clock::now() - inicio_gpu;

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        Barenn::Camada l1 = Barenn::criar_camada(Barenn::entradas, 32, false);
        Barenn::Camada l2 = Barenn::criar_camada(32, 16, false);
        Barenn::Camada l3 = Barenn::criar_camada(16, Barenn::classes, true);
        array<double, epocas> perdas{};
        const int n = static_cast<int>(treino.size());

        vector<float> h_imagens;
        vector<int> h_rotulos;
        Barenn::empacotar(treino, h_imagens, h_rotulos);
        float* d_imagens = nullptr;
        int* d_rotulos = nullptr;
        float* d_grad = nullptr;
        float* d_perdas = nullptr;  // perda de cada amostra em cada época
        CUDA_CHECK(cudaMalloc(&d_imagens, h_imagens.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rotulos, h_rotulos.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_grad, Barenn::classes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perdas,
                              static_cast<size_t>(epocas) * n * sizeof(float)));

        // A medição inclui a cópia do conjunto de treino e a volta das perdas.
        const auto inicio = chrono::steady_clock::now();
        CUDA_CHECK(cudaMemcpy(d_imagens, h_imagens.data(),
                              h_imagens.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rotulos, h_rotulos.data(),
                              h_rotulos.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        for (int epoca = 0; epoca < epocas; ++epoca) {
            float* perdas_epoca = d_perdas + static_cast<size_t>(epoca) * n;
            for (int k = 0; k < n; ++k) {
                Barenn::k_amostra<<<1, BLOCO_TREINO>>>(
                    l1, l2, l3, d_imagens, d_rotulos, k, d_grad, perdas_epoca, taxa);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        vector<float> h_perdas(static_cast<size_t>(epocas) * n);
        CUDA_CHECK(cudaMemcpy(h_perdas.data(), d_perdas,
                              h_perdas.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;
        // Soma em double na ordem das amostras, como na CPU.
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            for (int k = 0; k < n; ++k) {
                perda_total += h_perdas[static_cast<size_t>(epoca) * n + k];
            }
            perdas[epoca] = perda_total / n;
        }
        Barenn::relatorio_verificacao(perdas, l1, l2, l3);

        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";

        // Avaliação: todas as imagens de teste de uma vez, um thread por imagem.
        const auto inicio_avaliacao = chrono::steady_clock::now();
        const int m = static_cast<int>(teste.size());
        vector<float> h_teste;
        vector<int> h_rotulos_teste;
        Barenn::empacotar(teste, h_teste, h_rotulos_teste);
        float* d_teste = nullptr;
        float* d_escores = nullptr;
        CUDA_CHECK(cudaMalloc(&d_teste, h_teste.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_escores,
                              static_cast<size_t>(m) * Barenn::classes * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_teste, h_teste.data(), h_teste.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        Barenn::k_inferencia<<<(m + Barenn::bloco_inferencia - 1) /
                                   Barenn::bloco_inferencia,
                               Barenn::bloco_inferencia>>>(l1, l2, l3, d_teste, m,
                                                           d_escores);
        CUDA_CHECK(cudaGetLastError());
        vector<array<float, Barenn::classes>> escores(m);
        CUDA_CHECK(cudaMemcpy(escores.data(), d_escores,
                              static_cast<size_t>(m) * Barenn::classes * sizeof(float),
                              cudaMemcpyDeviceToHost));
        Barenn::relatorio_avaliacao(escores, teste);
        const chrono::duration<double> tempo_avaliacao =
            chrono::steady_clock::now() - inicio_avaliacao;

        cout << "Tempo de leitura " << tempo_leitura.count() << " s\n"
             << "Tempo de avaliacao " << tempo_avaliacao.count() << " s\n"
             << "Tempo de criacao do contexto CUDA " << tempo_contexto.count() << " s\n"
             << "# GPU: " << prop.name << " (sm_" << prop.major << prop.minor
             << ", " << prop.multiProcessorCount << " SMs), bloco de treino "
             << BLOCO_TREINO << " threads\n";

        CUDA_CHECK(cudaFree(d_imagens));
        CUDA_CHECK(cudaFree(d_rotulos));
        CUDA_CHECK(cudaFree(d_grad));
        CUDA_CHECK(cudaFree(d_perdas));
        CUDA_CHECK(cudaFree(d_teste));
        CUDA_CHECK(cudaFree(d_escores));
        Barenn::liberar_camada(l1);
        Barenn::liberar_camada(l2);
        Barenn::liberar_camada(l3);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
