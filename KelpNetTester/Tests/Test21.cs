using System;
using System.Diagnostics;
using System.Drawing;
using KelpNet.Common;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Noise;
using KelpNet.Functions.Normalization;
using KelpNet.Functions.Poolings;
using KelpNet.Loss;
using KelpNet.Optimizers;
using KelpNetTester.TestData;
using ReflectSoftware.Insight;

namespace KelpNetTester.Tests
{
    class Test21
    {
        const int BATCH_DATA_COUNT = 20;
        const int TRAIN_DATA_COUNT = 3000; // = 60000 / 20
        const int TEST_DATA_COUNT = 200;
        public static bool Passed = false;

        // MNIST accuracy tester
        public static void Run(double accuracyThreshold = .9979D)
        {
            MnistData mnistData = new MnistData(28);
            Real maxAccuracy = 0;
            //Number of middle layers
            const int N = 30; //It operates at 1000 similar to the reference link but it is slow at the CPU

            ReflectInsight ri = new ReflectInsight("Test21");
            ri.Enabled = true;
            RILogManager.Add("Test21", "Test21");
            RILogManager.SetDefault("Test21");

            //FunctionStack nn = new FunctionStack("Test21",
            //    new Linear(28 * 28, 1024, name: "l1 Linear"),
            //    new Sigmoid(name: "l1 Sigmoid"),
            //    new Linear(1024, 10, name: "l2 Linear")
            //);
            //nn.SetOptimizer(new MomentumSGD());

            FunctionStack nn = new FunctionStack("Test7",
    new Linear(true, 28 * 28, N, name: "l1 Linear"), // L1
                new BatchNormalization(true, N, name: "l1 BatchNorm"),
                new ReLU(name: "l1 ReLU"),
                new Linear(true, N, N, name: "l2 Linear"), // L2
                new BatchNormalization(true, N, name: "l2 BatchNorm"),
                new ReLU(name: "l2 ReLU"),
                new Linear(true, N, N, name: "l3 Linear"), // L3
                new BatchNormalization(true, N, name: "l3 BatchNorm"),
                new ReLU(name: "l3 ReLU"),
                new Linear(true, N, N, name: "l4 Linear"), // L4
                new BatchNormalization(true, N, name: "l4 BatchNorm"),
                new ReLU(name: "l4 ReLU"),
                new Linear(true, N, N, name: "l5 Linear"), // L5
                new BatchNormalization(true, N, name: "l5 BatchNorm"),
                new ReLU(name: "l5 ReLU"),
                new Linear(true, N, N, name: "l6 Linear"), // L6
                new BatchNormalization(true, N, name: "l6 BatchNorm"),
                new ReLU(name: "l6 ReLU"),
                new Linear(true, N, N, name: "l7 Linear"), // L7
                new BatchNormalization(true, N, name: "l7 BatchNorm"),
                new ReLU(name: "l7 ReLU"),
                new Linear(true, N, N, name: "l8 Linear"), // L8
                new BatchNormalization(true, N, name: "l8 BatchNorm"),
                new ReLU(name: "l8 ReLU"),
                new Linear(true, N, N, name: "l9 Linear"), // L9
                new BatchNormalization(true, N, name: "l9 BatchNorm"),
                new ReLU(name: "l9 ReLU"),
                new Linear(true, N, N, name: "l10 Linear"), // L10
                new BatchNormalization(true, N, name: "l10 BatchNorm"),
                new ReLU(name: "l10 ReLU"),
                new Linear(true, N, N, name: "l11 Linear"), // L11
                new BatchNormalization(true, N, name: "l11 BatchNorm"),
                new ReLU(name: "l11 ReLU"),
                new Linear(true, N, N, name: "l12 Linear"), // L12
                new BatchNormalization(true, N, name: "l12 BatchNorm"),
                new ReLU(name: "l12 ReLU"),
                new Linear(true, N, N, name: "l13 Linear"), // L13
                new BatchNormalization(true, N, name: "l13 BatchNorm"),
                new ReLU(name: "l13 ReLU"),
                new Linear(true, N, N, name: "l14 Linear"), // L14
                new BatchNormalization(true, N, name: "l14 BatchNorm"),
                new ReLU(name: "l14 ReLU"),
                new Linear(true, N, 10, name: "l15 Linear") // L15
            );

            // 0.0005 - 97.5, 0.001, 0.00146
            double alpha = 0.001;
            double beta1 = 0.9D;
            double beta2 = 0.999D;
            double epsilon = 1e-8;

            nn.SetOptimizer(new Adam("Adam21", alpha, beta1, beta2, epsilon));

            Stopwatch sw = new Stopwatch();
            sw.Start();

            for (int epoch = 0; epoch < 3; epoch++)
            {
                Real totalLoss = 0;
                long totalLossCount = 0;

                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    TestDataSet datasetX = mnistData.GetRandomXSet(BATCH_DATA_COUNT, 28, 28);
                    Real sumLoss = Trainer.Train(nn, datasetX.Data, datasetX.Label, new SoftmaxCrossEntropy());
                    totalLoss = sumLoss;
                    totalLossCount++;

                    if (i % 20 == 0)
                    {
                        TestDataSet datasetY = mnistData.GetRandomYSet(TEST_DATA_COUNT, 28);
                        Real accuracy = Trainer.Accuracy(nn, datasetY.Data, datasetY.Label, false);
                        if (accuracy > maxAccuracy)
                            maxAccuracy = accuracy;
                        Passed = (accuracy >= accuracyThreshold);

                        sw.Stop();
                        ri.ViewerSendWatch("Iteration", "epoch " + (epoch + 1) + " of 3, batch " + i + " of " + TRAIN_DATA_COUNT);
                        ri.ViewerSendWatch("Max Accuracy", maxAccuracy * 100 + "%");
                        ri.ViewerSendWatch("Current Accuracy", accuracy * 100 + "%");
                        ri.ViewerSendWatch("Total Loss ", totalLoss / totalLossCount);
                        ri.ViewerSendWatch("Elapsed Time", Helpers.FormatTimeSpan(sw.Elapsed));
                        ri.ViewerSendWatch("Accuracy Threshold", Passed ? "Passed" : "Not Passed");
                        sw.Start();
                    }
                }

                sw.Stop();
                ri.SendInformation("Total Processing Time: " + Helpers.FormatTimeSpan(sw.Elapsed));
            }
        }
    }
}
