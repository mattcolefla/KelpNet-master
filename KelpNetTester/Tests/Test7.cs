﻿using System;
using KelpNet.Common;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Normalization;
using KelpNet.Loss;
using KelpNet.Optimizers;
using KelpNetTester.TestData;

namespace KelpNetTester.Tests
{
    using ReflectSoftware.Insight;

    //Learning of MNIST (handwritten character) by 15-layer MLP using batch normalization
    //reference： http://takatakamanbou.hatenablog.com/entry/2015/12/20/233232
    class Test7
    {
        //Number of mini batches
        const int BATCH_DATA_COUNT = 128;

        //Number of exercises per generation
        const int TRAIN_DATA_COUNT = 1000;//50000;

        //
        const int TEST_DATA_COUNT = 200;

        //Number of middle layers
        const int N = 30; //It operates at 1000 similar to the reference link but it is slow at the CPU

        public static void Run()
        {
            RILogManager.Default?.SendDebug("MNIST Data Loading...");
            MnistData mnistData = new MnistData(28);

            RILogManager.Default?.SendDebug("Training Start...");

            //Writing the network configuration in FunctionStack
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

            nn.SetOptimizer(new AdaGrad());


            for (int epoch = 0; epoch < 3; epoch++)
            {
                Real totalLoss = 0;
                long totalLossCounter = 0;

                //Run the batch
                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    RILogManager.Default?.SendDebug("epoch " + (epoch + 1) + " of 3, Batch " + i + " of " + TRAIN_DATA_COUNT);

                    //Get data randomly from training data
                    TestDataSet datasetX = mnistData.GetRandomXSet(BATCH_DATA_COUNT, 28, 28);

                    //Learn
                    Real sumLoss = Trainer.Train(nn, datasetX.Data, datasetX.Label, new SoftmaxCrossEntropy());
                    totalLoss += sumLoss;
                    totalLossCounter++;

                    if (i % 20 == 0)
                    {
                        RILogManager.Default?.SendDebug("batch count " + i + "/" + TRAIN_DATA_COUNT);
                        RILogManager.Default?.SendDebug("total loss " + totalLoss / totalLossCounter);
                        RILogManager.Default?.SendDebug("local loss " + sumLoss);
                        
                        RILogManager.Default?.SendDebug("Testing random data...");

                        //Get data randomly from test data
                        TestDataSet datasetY = mnistData.GetRandomYSet(TEST_DATA_COUNT, 28);

                        //Run the test
                        Real accuracy = Trainer.Accuracy(nn, datasetY.Data, datasetY.Label);
                        RILogManager.Default?.SendDebug("Test Accuracy: " + accuracy);
                    }
                }
            }

            ModelIO.Save(nn, "Test7.nn");
            RILogManager.Default?.SendDebug(nn.Describe());
        }
    }
}
