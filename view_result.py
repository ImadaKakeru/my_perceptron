import cv2
import os
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import pandas as pd
import torch
import seaborn as sns
import numpy as np
def viewResult(images, GT):
    # 保存先フォルダがなければ作成
    if not os.path.exists("./resultImage"):
        os.makedirs("./resultImage")

    for i in range(len(images)):
        # # データ型と形状の確認
        # print("Original image shape:", images[i].shape)
        # print("Original image dtype:", images[i].dtype)

        # 最初の次元が1の場合、次元を取り除く
        image = images[i][0] if images[i].shape[0] == 1 else images[i]

        # 画像をuint8型に変換し、0-255の範囲で整数に
        image = (image * 255).astype("uint8") if image.max() <= 1 else image.astype("uint8")

        # 画像を4倍に拡大
        image = cv2.resize(image, (640, 480))  # (横, 縦)の順番で指定

        # もし1チャネルの場合は3チャネルに変換
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        #
        # print("Processed image shape:", image.shape)
        # print("Processed image dtype:", image.dtype)

        gt = GT[i]
        # print("gt:", gt)

        # 座標も4倍にスケールアップしてから描画
        cv2.rectangle(image, (int(gt[0]) - 40, int(gt[1]) - 40),
                             (int(gt[0] ) + 40, int(gt[1]) + 40),
                             (0, 255, 0), 5)

        cv2.imwrite(f"./resultImage/result{i}.png", image)

    return image

def Hakohige(data1, data2):
    # データをリスト形式でまとめて渡す
    data = [data1.numpy(), data2.numpy()]

    # 箱ひげ図を横並びで表示
    fig = plt.figure(figsize=(10, 6))
    # plt.boxplot(data, vert=False, patch_artist=True, labels=['Data1', 'Data2'])
    bax = brokenaxes(ylims=((0, 60), (180, 200)), hspace=0.1)  # y軸で省略する範囲を指定

    bax.boxplot(data, widths=0.6,vert=True, patch_artist=True, labels=['loss', 'l2'])

    # 波線を描画する
    # xlim = bax.get_xlim()
    # for i in range(2):
    #     bax[i].plot([0.1, 0.9], [1, 1], transform=bax[i].transAxes, color='black', linestyle='-', lw=2, marker='~', markersize=18)
    # # 中央値の表示（横方向に合わせてy軸に線を描画）
    # median_value1 = data1.median().item()
    # median_value2 = data2.median().item()
    # plt.axhline(1, color='red', linestyle='--', label=f'Data1 Median: {median_value1:.2f}')
    # plt.axhline(2, color='blue', linestyle='--', label=f'Data2 Median: {median_value2:.2f}')

    plt.title('Box Plot of Data')
    plt.xlabel('Values')
    # plt.legend(loc='upper right')
    plt.show()

def strip_plot(data1, data2):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # データをnumpy配列に変換
    data1 = data1.numpy()
    data2 = data2.numpy()

    # 1. 外れ値を含むデータのプロット
    means = [np.mean(data1), np.mean(data2)]
    std_devs = [np.std(data1), np.std(data2)]

    # 新しいFigureを作成
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    # 棒グラフ & ストリッププロット（重ねて表示）
    x_labels = ["Data1", "Data2"]
    ax1.bar(x_labels, means, yerr=std_devs, capsize=10, color=["lightcoral", "lightskyblue"], alpha=0.6)

    # DataFrameを作成してストリッププロットに渡す
    df = pd.DataFrame(
        {"Category": ["Data1"] * len(data1) + ["Data2"] * len(data2), "Values": np.concatenate([data1, data2])})
    sns.stripplot(data=df, x="Category", y="Values", ax=ax1, color="blue", jitter=True, alpha=0.7)
    ax1.set_title("Original Data: Mean and Standard Deviation")
    ax1.set_ylabel("Values")

    # y軸の範囲を指定
    ax1.set_ylim(-10, 130)

    # 2. 外れ値を除いたデータのプロット
    def remove_outliers(data):
        mean = np.mean(data)
        std_dev = np.std(data)
        return data[(data > mean - 2 * std_dev) & (data < mean + 2 * std_dev)]

    data1_no_outliers = remove_outliers(data1)
    data2_no_outliers = remove_outliers(data2)
    means_no_outliers = [np.mean(data1_no_outliers), np.mean(data2_no_outliers)]
    print(means_no_outliers)
    std_devs_no_outliers = [np.std(data1_no_outliers), np.std(data2_no_outliers)]
    print(std_devs_no_outliers)
    # 新しいFigureを作成
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    x_labels_no_outliers = ["Data1_no_outliers", "Data2_no_outliers"]
    ax2.bar(x_labels_no_outliers, means_no_outliers, yerr=std_devs_no_outliers, capsize=10,
            color=["lightcoral", "lightskyblue"], alpha=0.6)

    # DataFrameを作成してストリッププロットに渡す
    df_no_outliers = pd.DataFrame(
        {"Category": ["Data1_no_outliers"] * len(data1_no_outliers) + ["Data2_no_outliers"] * len(data2_no_outliers),
         "Values": np.concatenate([data1_no_outliers, data2_no_outliers])})
    sns.stripplot(data=df_no_outliers, x="Category", y="Values", ax=ax2, color="green", jitter=True, alpha=0.7)
    ax2.set_title("Without Outliers: Mean and Standard Deviation")
    ax2.set_ylabel("Values")

    # 同じy軸範囲を設定
    # ax2.set_ylim(-10, 200)
    ax2.set_ylim(-10, 130)

    # 図を表示・保存
    plt.show()
    fig1.savefig("./all_result.png")
    fig2.savefig("./no_outliers_result.png")