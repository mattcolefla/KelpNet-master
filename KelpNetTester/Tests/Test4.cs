using System;
using KelpNet.Common;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;
using KelpNetTester.TestData;

namespace KelpNetTester.Tests
{
    using ReflectSoftware.Insight;

    //Learning of MNIST (Handwritten Characters) by MLP
    class Test4
    {
        const int BATCH_DATA_COUNT = 20;
        const int TRAIN_DATA_COUNT = 3000; // = 60000 / 20
        const int TEST_DATA_COUNT = 200;


        public static void Run()
        {
            RILogManager.Default?.SendDebug("MNIST Data Loading...");
            RILogManager.Default?.SendDebug("MNIST Data Loading...");
            MnistData mnistData = new MnistData(28);

            RILogManager.Default?.SendDebug("Training Start...");

            FunctionStack nn = new FunctionStack("Test4",
                new Linear(true, 28 * 28, 1024, name: "l1 Linear"),
                new Sigmoid(name: "l1 Sigmoid"),
                new Linear(true, 1024, 10, name: "l2 Linear")
            );

            nn.SetOptimizer(new MomentumSGD());

            for (int epoch = 0; epoch < 3; epoch++)
            {
                RILogManager.Default?.SendDebug("epoch " + (epoch + 1));

                Real totalLoss = 0;
                long totalLossCount = 0;

                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    //Get data randomly from training data
                    TestDataSet datasetX = mnistData.GetRandomXSet(BATCH_DATA_COUNT, 28, 28);

                    Real sumLoss = Trainer.Train(nn, datasetX.Data, datasetX.Label, new SoftmaxCrossEntropy());
                    totalLoss = sumLoss;
                    totalLossCount++;

                    if (i % 20 == 0)
                    {
                        RILogManager.Default?.SendDebug("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);
                        RILogManager.Default?.SendDebug("total loss " + totalLoss / totalLossCount);
                        RILogManager.Default?.SendDebug("local loss " + sumLoss);

                        RILogManager.Default?.SendDebug("\nTesting...");

                        //Get data randomly from test data
                        TestDataSet datasetY = mnistData.GetRandomYSet(TEST_DATA_COUNT, 28);

                        Real accuracy = Trainer.Accuracy(nn, datasetY.Data, datasetY.Label);
                        RILogManager.Default?.SendDebug("accuracy " + accuracy);
                    }
                }
            }
        }
    }
}
