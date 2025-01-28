# CUDA LBM Simulation

## 概要
このプロジェクトは、CUDAを使用して格子ボルツマン法（LBM）を並列に実行するプログラムです。流体力学のシミュレーションを効率的に行うことができます。

## 導入手順
### 必要条件
- CUDA対応のGPU
- CUDA Toolkit
- Python 3.x
- 必要なPythonライブラリ（`numpy`, `matplotlib`, `pillow`）

### インストール
1. CUDA Toolkitをインストールします。詳細な手順は[CUDAの公式サイト](https://developer.nvidia.com/cuda-downloads)を参照してください。
2. 必要なPythonライブラリをインストールします。
    ```sh
    pip install numpy matplotlib pillow
    ```

## 実行方法
1. [setting.json](./setting.json)ファイルでシミュレーションの設定を行います。
2. 以下のコマンドを実行してシミュレーションを開始します。
    ```sh
    nvcc -o lbm_cuda lbm_cuda.cu
    ./lbm_cuda
    ```
3. シミュレーション結果は[result.json](./result.json)に保存されます。
4. [visualize.py](./visualize.py)を実行して結果をGIFとして可視化します。
    ```sh
    python visualize.py
    ```

## 依存関係
- CUDA Toolkit
- Python 3.x
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- `pillow`


## プログラムの説明
### [lbm_cuda.cu](./lbm_cuda.cu)
このファイルには、CUDAを使用してLBMシミュレーションを実行するためのコードが含まれています。主な関数は以下の通りです：
- `stream`: 流体の移動をシミュレートします。
- `collide`: 流体の衝突をシミュレートします。
- `load_config`: 設定ファイルからシミュレーションパラメータを読み込みます。
- `initialize_barrier`: バリアを設定します。
- `save_to_json`: シミュレーション結果をJSONファイルに保存します。

### [visualize.py](./visualize.py)
このファイルには、シミュレーション結果を読み込み、GIFとして可視化するためのコードが含まれています。主な関数は以下の通りです：
- `load_data_from_json`: JSONファイルからデータを読み込みます。
- `visualize_and_save_gif`: 結果を可視化してGIFに保存します。

### [setting.json](./setting.json)
シミュレーションの設定を行うためのファイルです。以下のパラメータを設定できます：
- `height`: シミュレーション領域の高さ
- `width`: シミュレーション領域の幅
- `viscosity`: 流体の粘性
- `u0`: 初期速度
- `total_steps`: シミュレーションの総ステップ数
- `skip_frames`: フレームをスキップする間隔
- `barrier`: バリアの位置情報

### [result.json](./result.json)
シミュレーション結果が保存されるファイルです。

## 参考文献
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Lattice Boltzmann Method](https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods)

## ライセンス
このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](./LICENSE)ファイルを参照してください。