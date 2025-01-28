import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# JSONファイルからデータを読み込む関数
def load_data_from_json(file_path):
    with open(file_path, 'r') as f:
        result = json.load(f)
    return result

# 結果を可視化してGIFに保存する関数
def visualize_and_save_gif(result, total_step, height, width, gif_name="simulation.gif"):
    fig, ax = plt.subplots()

    # 初期フレームを設定
    def init():
        im.set_array(np.zeros((height, width)))
        return [im]

    # フレームごとの更新処理
    def update(frame):
        # フレームのデータをfloatに変換
        data = np.array(result[frame], dtype=np.float32)
        im.set_array(data)
        return [im]

    # 最初のフレーム用のデータ
    initial_data = np.array(result[0], dtype=np.float32)
    im = ax.imshow(initial_data, cmap='viridis', interpolation='none')

    # アニメーション設定
    ani = animation.FuncAnimation(fig, update, frames=total_step, init_func=init, blit=True, repeat=False)

    # GIFとして保存（Pillowライターを使用）
    ani.save(gif_name, writer='pillow', fps=5)

    plt.close()  # プロットを閉じる

# メイン処理
if __name__ == "__main__":
    # JSONファイルのパス
    json_file_path = "result.json"
    output_file_name = "simulation.gif"

    # データを読み込む
    result = load_data_from_json(json_file_path)

    # シミュレーションのステップ数、画面の高さと幅
    total_step = len(result)
    height = len(result[0])
    width = len(result[0][0])

    # GIFアニメーションを作成し保存
    visualize_and_save_gif(result, total_step, height, width, gif_name=output_file_name)
    print(output_file_name+" saved!")
