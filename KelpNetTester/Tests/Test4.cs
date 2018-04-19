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
    //Learning of MNIST (Handwritten Characters) by MLP
    class Test4
    {
        const int BATCH_DATA_COUNT = 20;
        const int TRAIN_DATA_COUNT = 3000; // = 60000 / 20
        const int TEST_DATA_COUNT = 200;


        public static void Run()
        {
            Console.WriteLine("MNIST Data Loading...");
            MnistData mnistData = new MnistData(28);

            Console.WriteLine("Training Start...");

            FunctionStack nn = new FunctionStack(
                new Linear(28 * 28, 1024, name: "l1 Linear"),
                new Sigmoid(name: "l1 Sigmoid"),
                new Linear(1024, 10, name: "l2 Linear")
            );

            nn.SetOptimizer(new MomentumSGD());

            for (int epoch = 0; epoch < 3; epoch++)
            {
                Console.WriteLine("epoch " + (epoch + 1));

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
                        Console.WriteLine("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);
                        Console.WriteLine("total loss " + totalLoss / totalLossCount);
                        Console.WriteLine("local loss " + sumLoss);

                        Console.WriteLine("\nTesting...");

                        //Get data randomly from test data
                        TestDataSet datasetY = mnistData.GetRandomYSet(TEST_DATA_COUNT, 28);

                        Real accuracy = Trainer.Accuracy(nn, datasetY.Data, datasetY.Label);
                        Console.WriteLine("accuracy " + accuracy);
                    }
                }
            }
        }
    }
}
