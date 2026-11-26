// Barenn em CUDA - porting inicial (v1.0).
//
// Mesma rede (64 -> 32 -> 16 -> 10), mesma inicialização, mesmo SGD por amostra
// e mesma ordem de cada soma da versão CPU. Compilado com --fmad=false para que
// a*b+c não vire FMA: assim o resultado é bit a bit igual ao do legado.
//
// Estratégia deste porting inicial (propositalmente direta):
//   - cada camada tem kernels separados de forward, delta, gradiente da entrada
//     e atualização dos pesos; um thread por neurônio / por peso;
//   - a cada amostra, a imagem é copiada Host->Device e a perda volta
//     Device->Host (duas transferências PCIe e ~12 lançamentos por amostra).
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

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Um thread por neurônio de saída: z = b + soma_j x[j] * w[j][i], na ordem j.
__global__ void k_forward(const float* x, const float* wt, const float* b,
                          float* z, float* out, int in, int nout, bool final) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nout) return;
    float sum = b[i];
    for (int j = 0; j < in; ++j) {
        sum += x[j] * wt[j * nout + i];
    }
    z[i] = sum;
    // Mesma expressão de std::max(0.0f, sum).
    out[i] = final ? sum : ((0.0f < sum) ? sum : 0.0f);
}

// Metade da soma dos erros quadráticos e o gradiente por escore (um thread).
__global__ void k_perda(const float* escores, int rotulo, float* grad,
                        float* perda) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        grad[i] = erro;
    }
    *perda = total;
}

// Delta: gradiente em relação à soma antes da ativação (ReLU nas ocultas).
__global__ void k_delta(const float* grad_out, const float* z, float* delta,
                        int nout, bool final) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nout) return;
    delta[i] = final ? grad_out[i] : grad_out[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
}

// Um thread por entrada: grad_in[j] = soma_i delta[i] * w[j][i], na ordem i.
// Roda antes da atualização, com os pesos antigos.
__global__ void k_grad_entrada(const float* delta, const float* wt,
                               float* grad_in, int in, int nout) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= in) return;
    float sum = 0;
    for (int i = 0; i < nout; ++i) {
        sum += delta[i] * wt[j * nout + i];
    }
    grad_in[j] = sum;
}

// Um thread por peso: w[j][i] -= lr * (delta[i] * x[j]); os primeiros nout
// threads também atualizam o viés.
__global__ void k_atualiza(float* wt, float* b, const float* delta,
                           const float* x, int in, int nout, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < in * nout) {
        int j = idx / nout;
        int i = idx % nout;
        wt[idx] -= lr * (delta[i] * x[j]);
    }
    if (idx < nout) {
        b[idx] -= lr * delta[idx];
    }
}

// ---------------------------------------------------------------------------
// Camada: parâmetros e buffers no device
// ---------------------------------------------------------------------------
class CamadaGPU {
    int in_features;
    int out_features;
    bool final;
    bool propaga_gradiente;
    const float* d_entrada = nullptr;  // entrada do último forward (no device)

public:
    float* d_wt = nullptr;     // pesos transpostos [entrada][saída]
    float* d_b = nullptr;      // viés
    float* d_z = nullptr;      // soma antes da ativação
    float* d_out = nullptr;    // saída após ativação
    float* d_delta = nullptr;
    float* d_grad_in = nullptr;

    CamadaGPU(int input_size, int output_size, bool is_final, bool propaga)
        : in_features(input_size), out_features(output_size), final(is_final),
          propaga_gradiente(propaga) {
        // Mesma inicialização da CPU: semente 1337, ordem lógica (i, j).
        vector<float> wt(static_cast<size_t>(in_features) * out_features);
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (int i = 0; i < out_features; ++i) {
            for (int j = 0; j < in_features; ++j) {
                wt[static_cast<size_t>(j) * out_features + i] = distribution(gen);
            }
        }
        vector<float> b(out_features, 0.0f);
        CUDA_CHECK(cudaMalloc(&d_wt, wt.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b, out_features * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_z, out_features * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, out_features * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_delta, out_features * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_in, in_features * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_wt, wt.data(), wt.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b.data(), b.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    ~CamadaGPU() {
        cudaFree(d_wt);
        cudaFree(d_b);
        cudaFree(d_z);
        cudaFree(d_out);
        cudaFree(d_delta);
        cudaFree(d_grad_in);
    }

    CamadaGPU(const CamadaGPU&) = delete;
    CamadaGPU& operator=(const CamadaGPU&) = delete;

    const float* Forward(const float* d_x) {
        d_entrada = d_x;
        k_forward<<<1, out_features>>>(d_x, d_wt, d_b, d_z, d_out, in_features,
                                       out_features, final);
        return d_out;
    }

    const float* Backward(const float* d_grad_out, float lr) {
        k_delta<<<1, out_features>>>(d_grad_out, d_z, d_delta, out_features, final);
        if (propaga_gradiente) {
            k_grad_entrada<<<1, in_features>>>(d_delta, d_wt, d_grad_in,
                                               in_features, out_features);
        }
        const int total = in_features * out_features;
        const int bloco = 256;
        k_atualiza<<<(total + bloco - 1) / bloco, bloco>>>(
            d_wt, d_b, d_delta, d_entrada, in_features, out_features, lr);
        return d_grad_in;
    }

    // Resumo dos parâmetros (ordem lógica [saída][entrada], depois o viés),
    // idêntico ao da versão CPU, para o modo de verificação.
    void Resumo(uint64_t& hash, double& soma) const {
        vector<float> wt(static_cast<size_t>(in_features) * out_features);
        vector<float> b(out_features);
        CUDA_CHECK(cudaMemcpy(wt.data(), d_wt, wt.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), d_b, b.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
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
            mistura(b[i]);
        }
    }
};

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
void relatorio_verificacao(const array<double, N>& perdas, const CamadaGPU& l1,
                           const CamadaGPU& l2, const CamadaGPU& l3) {
    if (!getenv("BARENN_VERIFICAR")) {
        return;
    }
    cerr << hexfloat;
    for (size_t epoca = 0; epoca < N; ++epoca) {
        cerr << "perda " << epoca + 1 << " " << perdas[epoca] << "\n";
    }
    const CamadaGPU* camadas[] = {&l1, &l2, &l3};
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
        Barenn::CamadaGPU l1(Barenn::entradas, 32, false, false);
        Barenn::CamadaGPU l2(32, 16, false, true);
        Barenn::CamadaGPU l3(16, Barenn::classes, true, true);
        array<double, epocas> perdas{};

        float* d_x = nullptr;
        float* d_grad = nullptr;
        float* d_perda = nullptr;
        CUDA_CHECK(cudaMalloc(&d_x, Barenn::entradas * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad, Barenn::classes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perda, sizeof(float)));

        // A medição inclui as transferências por amostra, como no laço original.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            for (const auto& amostra : treino) {
                CUDA_CHECK(cudaMemcpy(d_x, amostra.imagem.data(),
                                      Barenn::entradas * sizeof(float),
                                      cudaMemcpyHostToDevice));
                const float* escores = l3.Forward(l2.Forward(l1.Forward(d_x)));
                Barenn::k_perda<<<1, 1>>>(escores, amostra.rotulo, d_grad, d_perda);
                const float* grad3 = l3.Backward(d_grad, taxa);
                const float* grad2 = l2.Backward(grad3, taxa);
                l1.Backward(grad2, taxa);
                CUDA_CHECK(cudaGetLastError());
                float perda_amostra;
                CUDA_CHECK(cudaMemcpy(&perda_amostra, d_perda, sizeof(float),
                                      cudaMemcpyDeviceToHost));
                perda_total += perda_amostra;
            }
            perdas[epoca] = perda_total / treino.size();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;
        Barenn::relatorio_verificacao(perdas, l1, l2, l3);

        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";

        // Avaliação: forward de cada imagem na GPU, escores de volta ao host.
        const auto inicio_avaliacao = chrono::steady_clock::now();
        vector<array<float, Barenn::classes>> escores(teste.size());
        for (size_t i = 0; i < teste.size(); ++i) {
            CUDA_CHECK(cudaMemcpy(d_x, teste[i].imagem.data(),
                                  Barenn::entradas * sizeof(float),
                                  cudaMemcpyHostToDevice));
            const float* d_escores = l3.Forward(l2.Forward(l1.Forward(d_x)));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(escores[i].data(), d_escores,
                                  Barenn::classes * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
        Barenn::relatorio_avaliacao(escores, teste);
        const chrono::duration<double> tempo_avaliacao =
            chrono::steady_clock::now() - inicio_avaliacao;

        cout << "Tempo de leitura " << tempo_leitura.count() << " s\n"
             << "Tempo de avaliacao " << tempo_avaliacao.count() << " s\n"
             << "Tempo de criacao do contexto CUDA " << tempo_contexto.count() << " s\n"
             << "# GPU: " << prop.name << " (sm_" << prop.major << prop.minor
             << ", " << prop.multiProcessorCount << " SMs)\n";

        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_grad));
        CUDA_CHECK(cudaFree(d_perda));
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
