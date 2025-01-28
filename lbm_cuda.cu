#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "json.hpp" // JSONライブラリを使用

#define debug(var)                  \
    do                              \
    {                               \
        std::cout << #var << " : "; \
        view(var);                  \
    } while (0)
template <typename T>
void view(T e) { std::cout << e << std::endl; }
template <typename T>
void view(const std::vector<T> &v)
{
    for (const auto &e : v)
    {
        std::cout << e << " ";
    }
    std::cout << std::endl;
}
template <typename T>
void view(const std::vector<std::vector<T>> &vv)
{
    for (const auto &v : vv)
    {
        view(v);
    }
}

#define four9ths (4.0 / 9.0)
#define one9th (1.0 / 9.0)
#define one36th (1.0 / 36.0)

__global__ void stream(double *nN, double *nS, double *nE, double *nW, double *nNE, double *nNW, double *nSE, double *nSW, bool *barrier, int height, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height * width)
        return;

    int x = idx % width;
    int y = idx / width;

    // Streaming step
    nN[idx] = nN[(y - 1 + height) % height * width + x];
    nS[idx] = nS[(y + 1) % height * width + x];
    nE[idx] = nE[y * width + (x + 1) % width];
    nW[idx] = nW[y * width + (x - 1 + width) % width];
    nNE[idx] = nNE[((y - 1 + height) % height) * width + (x + 1) % width];
    nNW[idx] = nNW[((y - 1 + height) % height) * width + (x - 1 + width) % width];
    nSE[idx] = nSE[((y + 1) % height) * width + (x + 1) % width];
    nSW[idx] = nSW[((y + 1) % height) * width + (x - 1 + width) % width];

    // Bounce-back for barriers
    if (barrier[idx])
    {
        nN[idx] = nS[idx];
        nS[idx] = nN[idx];
        nE[idx] = nW[idx];
        nW[idx] = nE[idx];
        nNE[idx] = nSW[idx];
        nNW[idx] = nSE[idx];
        nSE[idx] = nNW[idx];
        nSW[idx] = nNE[idx];
    }
}

__global__ void collide(double *n0, double *nN, double *nS, double *nE, double *nW, double *nNE, double *nNW, double *nSE, double *nSW, double *rho, double *ux, double *uy, int height, int width, double omega)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height * width)
        return;

    int i = idx / width;
    int j = idx % width;

    // Compute macroscopic quantities
    rho[idx] = n0[idx] + nN[idx] + nS[idx] + nE[idx] + nW[idx] + nNE[idx] + nSE[idx] + nNW[idx] + nSW[idx];
    ux[idx] = (nE[idx] + nNE[idx] + nSE[idx] - nW[idx] - nNW[idx] - nSW[idx]) / rho[idx];
    uy[idx] = (nN[idx] + nNE[idx] + nNW[idx] - nS[idx] - nSE[idx] - nSW[idx]) / rho[idx];

    double ux2 = ux[idx] * ux[idx];
    double uy2 = uy[idx] * uy[idx];
    double u2 = ux2 + uy2;
    double omu215 = 1 - 1.5 * u2;
    double uxuy = ux[idx] * uy[idx];
    double u0 = 0.1;

    // Collision step
    n0[idx] = (1 - omega) * n0[idx] + omega * four9ths * rho[idx] * omu215;
    nN[idx] = (1 - omega) * nN[idx] + omega * one9th * rho[idx] * (omu215 + 3 * uy[idx] + 4.5 * uy2);
    nS[idx] = (1 - omega) * nS[idx] + omega * one9th * rho[idx] * (omu215 - 3 * uy[idx] + 4.5 * uy2);
    nE[idx] = (1 - omega) * nE[idx] + omega * one9th * rho[idx] * (omu215 + 3 * ux[idx] + 4.5 * ux2);
    nW[idx] = (1 - omega) * nW[idx] + omega * one9th * rho[idx] * (omu215 - 3 * ux[idx] + 4.5 * ux2);
    nNE[idx] = (1 - omega) * nNE[idx] + omega * one36th * rho[idx] * (omu215 + 3 * (ux[idx] + uy[idx]) + 4.5 * (u2 + 2 * uxuy));
    nNW[idx] = (1 - omega) * nNW[idx] + omega * one36th * rho[idx] * (omu215 + 3 * (-ux[idx] + uy[idx]) + 4.5 * (u2 - 2 * uxuy));
    nSE[idx] = (1 - omega) * nSE[idx] + omega * one36th * rho[idx] * (omu215 + 3 * (ux[idx] - uy[idx]) + 4.5 * (u2 - 2 * uxuy));
    nSW[idx] = (1 - omega) * nSW[idx] + omega * one36th * rho[idx] * (omu215 + 3 * (-ux[idx] - uy[idx]) + 4.5 * (u2 + 2 * uxuy));

    // Boundary conditions for forced flow
    if (j == 0)
    { // Top boundary
        nE[idx] = (1.0 / 9.0) * (1 + 3 * u0 + 4.5 * u0 * u0 - 1.5 * u0 * u0);
        nNE[idx] = (1.0 / 36.0) * (1 + 3 * u0 + 4.5 * u0 * u0 - 1.5 * u0 * u0);
        nSE[idx] = (1.0 / 36.0) * (1 + 3 * u0 + 4.5 * u0 * u0 - 1.5 * u0 * u0);
    }
    else if (j == width - 1)
    { // Bottom boundary
        nW[idx] = (1.0 / 9.0) * (1 - 3 * u0 + 4.5 * u0 * u0 - 1.5 * u0 * u0);
        nNW[idx] = (1.0 / 36.0) * (1 - 3 * u0 + 4.5 * u0 * u0 - 1.5 * u0 * u0);
        nSW[idx] = (1.0 / 36.0) * (1 - 3 * u0 + 4.5 * u0 * u0 - 1.5 * u0 * u0);
    }


}

void load_config(const std::string &filename, int &height, int &width, double &viscosity, double &u0, int &total_steps, int &skip_frames, std::vector<std::tuple<int, int, int>> &barriers)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open config file");
    }

    // JSONファイルの内容を解析
    nlohmann::json config;
    file >> config;

    // 必要なパラメータをロード
    height = config["height"];
    width = config["width"];
    viscosity = config["viscosity"];
    u0 = config["u0"];
    total_steps = config["total_steps"];
    skip_frames = config["skip_frames"];


    // バリア情報を読み込む
    for (const auto &barrier : config["barrier"]) // 修正点: config["barriers"] → config["barrier"]
    {
        int start_row = barrier["start_row"]; // 修正点: barrier[0]["start_row"] → barrier["start_row"]
        int end_row = barrier["end_row"];     // 修正点: barrier[0]["end_row"] → barrier["end_row"]
        int col = barrier["col"];             // 修正点: barrier[0]["col"] → barrier["col"]
        barriers.emplace_back(start_row, end_row, col);
    }
}

// グリッドの初期化とバリア設定
void initialize_barrier(bool *barrier, std::vector<std::tuple<int, int, int>> &barriers, int height, int width)
{
    // バリアを設定
    for (int k = 0; k < barriers.size(); k++)
    {
        int start_row = std::get<0>(barriers[k]);
        int end_row = std::get<1>(barriers[k]);
        int col = std::get<2>(barriers[k]);

        for (int i = start_row; i < end_row; i++)
        {
            barrier[i * width + col] = true;
        }
    }
}

void save_to_json(const std::vector<std::vector<double>> &data, const std::string &filename, int height, int width)
{
    using json = nlohmann::json;
    json result;

    // フレームごとにデータをJSONに格納
    for (size_t frame = 0; frame < data.size(); frame++)
    {
        for (size_t i = 0; i < height; i++)
        {
            for (size_t j = 0; j < width; j++)
            {
                int idx = i * width + j;
                result[frame][i][j] = data[frame][idx];
            }
        }
    }

    // JSONファイルに書き込み
    std::ofstream file(filename);
    file << result.dump(4); // インデントを付けて書き込む
    file.close();
}

int main()
{
    // 計測開始
    clock_t start = clock();
    
    // 変数宣言
    int height, width, total_steps, skip_frames;
    double viscosity, u0;
    std::vector<std::tuple<int, int, int>> barriers;
    // 設定ファイルの読み込み
    load_config("setting.json", height, width, viscosity, u0, total_steps, skip_frames, barriers);
    // omegaの計算
    const double omega = 1.0 / (3.0 * viscosity + 0.5);
    std::cout << "omega:" << omega << std::endl;

    // Initialize arrays
    double *n0, *nN, *nS, *nE, *nW, *nNE, *nNW, *nSE, *nSW, *rho, *ux, *uy;
    bool *barrier;
    cudaMallocManaged(&n0, height * width * sizeof(double));
    cudaMallocManaged(&nN, height * width * sizeof(double));
    cudaMallocManaged(&nS, height * width * sizeof(double));
    cudaMallocManaged(&nE, height * width * sizeof(double));
    cudaMallocManaged(&nW, height * width * sizeof(double));
    cudaMallocManaged(&nNE, height * width * sizeof(double));
    cudaMallocManaged(&nNW, height * width * sizeof(double));
    cudaMallocManaged(&nSE, height * width * sizeof(double));
    cudaMallocManaged(&nSW, height * width * sizeof(double));
    cudaMallocManaged(&rho, height * width * sizeof(double));
    cudaMallocManaged(&ux, height * width * sizeof(double));
    cudaMallocManaged(&uy, height * width * sizeof(double));
    cudaMallocManaged(&barrier, height * width * sizeof(bool));

    // Initialize particle densities and barriers
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = y * width + x;
            n0[idx] = four9ths * (1 - 1.5 * u0 * u0);
            nN[idx] = one9th * (1 - 1.5 * u0 * u0);
            nS[idx] = one9th * (1 - 1.5 * u0 * u0);
            nE[idx] = one9th * (1 + 3 * u0 + 4.5 * u0 * u0 - 1.5 * u0 * u0);
            nW[idx] = one9th * (1 - 3 * u0 + 4.5 * u0 * u0 - 1.5 * u0 * u0);
            nNE[idx] = one36th * (1 + 3 * u0 + 4.5 * u0 * u0 - 1.5 * u0 * u0);
            nSE[idx] = one36th * (1 + 3 * u0 + 4.5 * u0 * u0 - 1.5 * u0 * u0);
            nNW[idx] = one36th * (1 - 3 * u0 + 4.5 * u0 * u0 - 1.5 * u0 * u0);
            nSW[idx] = one36th * (1 - 3 * u0 + 4.5 * u0 * u0 - 1.5 * u0 * u0);
            barrier[idx] = false;
        }
    }

    // Set barrier
    view("バリア設定スタート");
    initialize_barrier(barrier, barriers, height, width);

    // Main simulation loop
    std::vector<std::vector<double>> results;
    for (int step = 0; step < total_steps; ++step)
    {
        stream<<<(height * width + 255) / 256, 256>>>(nN, nS, nE, nW, nNE, nNW, nSE, nSW, barrier, height, width);
        collide<<<(height * width + 255) / 256, 256>>>(n0, nN, nS, nE, nW, nNE, nNW, nSE, nSW, rho, ux, uy, height, width, omega);
        cudaDeviceSynchronize();

        if (step % skip_frames == 0)
        {
            std::vector<double> frame(height * width);
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    frame[y * width + x] = rho[y * width + x];
                }
            }
            results.push_back(frame);
        }
    }

    // Save results to JSON
    save_to_json(results, "result.json", height, width);

    // Free memory
    cudaFree(n0);
    cudaFree(nN);
    cudaFree(nS);
    cudaFree(nE);
    cudaFree(nW);
    cudaFree(nNE);
    cudaFree(nNW);
    cudaFree(nSE);
    cudaFree(nSW);
    cudaFree(rho);
    cudaFree(ux);
    cudaFree(uy);
    cudaFree(barrier);

    return 0;
}