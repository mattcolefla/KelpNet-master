using System;
using System.Diagnostics;
using KelpNet.Common;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Noise;
using KelpNet.Functions.Poolings;
using KelpNet.Loss;
using KelpNet.Optimizers;
using KelpNetTester.TestData;

namespace KelpNetTester.Tests
{
    using ReflectSoftware.Insight;

    class Test18
    {
        //ミニバッチの数
        const int BATCH_DATA_COUNT = 20;

        //一世代あたりの訓練回数
        const int TRAIN_DATA_COUNT = 3000; // = 60000 / 20

        //性能評価時のデータ数
        const int TEACH_DATA_COUNT = 200;

        public static void Run()
        {
            Stopwatch sw = new Stopwatch();

            //MNISTのデータを用意する
            RILogManager.Default?.SendDebug("CIFAR Data Loading...");
            CifarData cifarData = new CifarData();

            //ネットワークの構成を FunctionStack に書き連ねる
            FunctionStack nn = new FunctionStack(
                new Convolution2D(3, 32, 3, name: "l1 Conv2D", gpuEnable: true),
                new ReLU(name: "l1 ReLU"),
                new MaxPooling(2, name: "l1 MaxPooling", gpuEnable: true),
                new Dropout(0.25, name: "l1 DropOut"),
                new Convolution2D(32, 64, 3, name: "l2 Conv2D", gpuEnable: true),
                new ReLU(name: "l2 ReLU"),
                new MaxPooling(2, 2, name: "l2 MaxPooling", gpuEnable: true),
                new Dropout(0.25, name: "l2 DropOut"),
                new Linear(13 * 13 * 64, 512, name: "l3 Linear", gpuEnable: true),
                new ReLU(name: "l3 ReLU"),
                new Dropout(name: "l3 DropOut"),
                new Linear(512, 10, name: "l4 Linear", gpuEnable: true)
            );

            //optimizerを宣言
            nn.SetOptimizer(new Adam());

            RILogManager.Default?.SendDebug("Training Start...");

            //三世代学習
            for (int epoch = 1; epoch < 3; epoch++)
            {
                RILogManager.Default?.SendDebug("epoch " + epoch);

                //全体での誤差を集計
                Real totalLoss = 0;
                long totalLossCount = 0;

                //何回バッチを実行するか
                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    sw.Restart();

                    RILogManager.Default?.SendDebug("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);

                    //訓練データからランダムにデータを取得
                    TestData.TestDataSet datasetX = cifarData.GetRandomXSet(BATCH_DATA_COUNT);

                    //バッチ学習を並列実行する
                    Real sumLoss = Trainer.Train(nn, datasetX.Data, datasetX.Label, new SoftmaxCrossEntropy());
                    totalLoss += sumLoss;
                    totalLossCount++;

                    //結果出力
                    RILogManager.Default?.SendDebug("total loss " + totalLoss / totalLossCount);
                    RILogManager.Default?.SendDebug("local loss " + sumLoss);

                    sw.Stop();
                    RILogManager.Default?.SendDebug("time" + sw.Elapsed.TotalMilliseconds);

                    //20回バッチを動かしたら精度をテストする
                    if (i % 20 == 0)
                    {
                        RILogManager.Default?.SendDebug("\nTesting...");

                        //テストデータからランダムにデータを取得
                        TestData.TestDataSet datasetY = cifarData.GetRandomYSet(TEACH_DATA_COUNT);

                        //テストを実行
                        Real accuracy = Trainer.Accuracy(nn, datasetY.Data, datasetY.Label);
                        RILogManager.Default?.SendDebug("accuracy " + accuracy);
                    }
                }
            }
        }
    }
}
