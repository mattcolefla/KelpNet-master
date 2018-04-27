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

    // Learning of MNIST (handwritten character) by 5-layer CNN
    // The only difference from Test 4 is network configuration and Optimizer
    class Test6
    {
        const int BATCH_DATA_COUNT = 20;
        const int TRAIN_DATA_COUNT = 3000; // = 60000 / 20
        const int TEACH_DATA_COUNT = 200;

        public static void Run()
        {
            Stopwatch sw = new Stopwatch();

            RILogManager.Default?.SendDebug("MNIST Data Loading...");
            MnistData mnistData = new MnistData(28);

            FunctionStack nn = new FunctionStack(
                new Convolution2D(1, 32, 5, pad: 2, name: "l1 Conv2D", gpuEnable: true),
                new ReLU(name: "l1 ReLU"),
                new MaxPooling(2, 2, name: "l1 MaxPooling", gpuEnable: true),
                new Convolution2D(32, 64, 5, pad: 2, name: "l2 Conv2D", gpuEnable: true),
                new ReLU(name: "l2 ReLU"),
                new MaxPooling(2, 2, name: "l2 MaxPooling", gpuEnable: true),
                new Linear(7 * 7 * 64, 1024, name: "l3 Linear", gpuEnable: true),
                new ReLU(name: "l3 ReLU"),
                new Dropout(name: "l3 DropOut"),
                new Linear(1024, 10, name: "l4 Linear", gpuEnable: true)
            );

            nn.SetOptimizer(new Adam());

            RILogManager.Default?.SendDebug("Training Start...");
            for (int epoch = 1; epoch < 3; epoch++)
            {
                RILogManager.Default?.SendDebug("epoch " + epoch);
                Real totalLoss = 0;
                long totalLossCount = 0;

                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    sw.Restart();

                    RILogManager.Default?.SendDebug("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);

                    TestDataSet datasetX = mnistData.GetRandomXSet(BATCH_DATA_COUNT, 28, 28);
                    Real sumLoss = Trainer.Train(nn, datasetX.Data, datasetX.Label, new SoftmaxCrossEntropy());
                    totalLoss += sumLoss;
                    totalLossCount++;

                    RILogManager.Default?.SendDebug("total loss " + totalLoss / totalLossCount);
                    RILogManager.Default?.SendDebug("local loss " + sumLoss);

                    sw.Stop();
                    RILogManager.Default?.SendDebug("time" + sw.Elapsed.TotalMilliseconds);

                    if (i % 20 == 0)
                    {
                        RILogManager.Default?.SendDebug("\nTesting...");
                        TestDataSet datasetY = mnistData.GetRandomYSet(TEACH_DATA_COUNT, 28);
                        Real accuracy = Trainer.Accuracy(nn, datasetY.Data, datasetY.Label);
                        RILogManager.Default?.SendDebug("accuracy " + accuracy);
                    }
                }
            }
        }
    }
}
