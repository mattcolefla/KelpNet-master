using System;
using System.Collections.Generic;
using System.Linq;


using KelpNet.Common;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Normalization;
using KelpNet.Loss;
using KelpNet.Optimizers;
using KelpNetTester.TestData;
using MathNet.Numerics.Statistics;


namespace KelpNetTester.Tests
{
    using KelpNet.Common.Functions;
    using ReflectSoftware.Insight;

    class Test20
    {
        // creating a massive deep learning neural network
        const int BATCH_DATA_COUNT = 128;
        private const int TRAIN_DATA_COUNT = 50000;
        const int TEST_DATA_COUNT = 200;
        private const int N = 30;
        private const int numLayers = 1000000;

        public static void Run()
        {
            int neuronCount = 28;
            Console.WriteLine("MNIST Data Loading...");
            MnistData mnistData = new MnistData(neuronCount);
            Console.WriteLine("Training Start...");

            FunctionStack nn = new FunctionStack();
            List<Function> functions = new List<Function>();
            for (int x = 1; x <= numLayers; x++)
            {
                functions.Add(new Linear(neuronCount * neuronCount, N, name: $"l{x} Linear"));
                functions.Add(new BatchNormalization(N, name: $"l{x} BatchNorm"));

                //if (x <= 5000)
                //{
                //    functions.Add(new ReLU(name: $"l{x} ReLU"));
                //}
                //else
                //{
                    functions.Add(new PolynomialApproximantSteep(slope: 0.000001, name: $"l{x} PolynomialApproximantSteep"));
                //}
            }

            nn.Add(new Linear(N, 10, noBias: false, name: $"l{numLayers + 1} Linear"));

            //FunctionStack nn = new FunctionStack(
            //    new Linear(neuronCount * neuronCount, N, name: "l1 Linear"), // L1
            //    new BatchNormalization(N, name: "l1 BatchNorm"),
            //    new LeakyReLU(slope: 0.000001, name: "l1 LeakyReLU"),
            //    new Linear(N, N, name: "l2 Linear"), // L2
            //    new BatchNormalization(N, name: "l2 BatchNorm"),
            //    new LeakyReLU(slope: 0.000001, name: "l2 LeakyReLU"),
            //    new Linear(N, N, name: "l3 Linear"), // L3
            //    new BatchNormalization(N, name: "l3 BatchNorm"),
            //    new LeakyReLU(slope: 0.000001, name: "l3 LeakyReLU"),
            //    new Linear(N, N, name: "l4 Linear"), // L4
            //    new BatchNormalization(N, name: "l4 BatchNorm"),
            //    new LeakyReLU(slope: 0.000001, name: "l4 LeakyReLU"),
            //    new Linear(N, N, name: "l5 Linear"), // L5
            //    new BatchNormalization(N, name: "l5 BatchNorm"),
            //    new LeakyReLU(slope: 0.000001, name: "l5 LeakyReLU"),
            //    new Linear(N, N, name: "l6 Linear"), // L6
            //    new BatchNormalization(N, name: "l6 BatchNorm"),
            //    new LeakyReLU(slope: 0.000001, name: "l6 LeakyReLU"),
            //    new Linear(N, N, name: "l7 Linear"), // L7
            //    new BatchNormalization(N, name: "l7 BatchNorm"),
            //    new LeakyReLU(slope: 0.000001, name: "l7 ReLU"),
            //    new Linear(N, N, name: "l8 Linear"), // L8
            //    new BatchNormalization(N, name: "l8 BatchNorm"),
            //    new LeakyReLU(slope: 0.000001, name: "l8 LeakyReLU"),
            //    new Linear(N, N, name: "l9 Linear"), // L9
            //    new BatchNormalization(N, name: "l9 BatchNorm"),
            //    new PolynomialApproximantSteep(slope: 0.000001, name: "l9 PolynomialApproximantSteep"),
            //    new Linear(N, N, name: "l10 Linear"), // L10
            //    new BatchNormalization(N, name: "l10 BatchNorm"),
            //    new PolynomialApproximantSteep(slope: 0.000001, name: "l10 PolynomialApproximantSteep"),
            //    new Linear(N, N, name: "l11 Linear"), // L11
            //    new BatchNormalization(N, name: "l11 BatchNorm"),
            //    new PolynomialApproximantSteep(slope: 0.000001, name: "l11 PolynomialApproximantSteep"),
            //    new Linear(N, N, name: "l12 Linear"), // L12
            //    new BatchNormalization(N, name: "l12 BatchNorm"),
            //    new PolynomialApproximantSteep(slope: 0.000001, name: "l12 PolynomialApproximantSteep"),
            //    new Linear(N, N, name: "l13 Linear"), // L13
            //    new BatchNormalization(N, name: "l13 BatchNorm"),
            //    new PolynomialApproximantSteep(slope: 0.000001, name: "l13 PolynomialApproximantSteep"),
            //    new Linear(N, N, name: "l14 Linear"), // L14
            //    new BatchNormalization(N, name: "l14 BatchNorm"),
            //    new PolynomialApproximantSteep(slope: 0.000001, name: "l14 PolynomialApproximantSteep"),
            //    new Linear(N, 10, name: "l15 Linear") // L15
            //);
            nn.SetOptimizer(new AdaGrad());

            RunningStatistics stats = new RunningStatistics();
            Histogram lossHistogram = new Histogram();
            Histogram accuracyHistogram = new Histogram();
            Real totalLoss = 0;
            long totalLossCounter = 0;
            Real highestAccuracy = 0;
            Real bestLocalLoss = 0;
            Real bestTotalLoss = 0;

            for (int epoch = 0; epoch < 1; epoch++)
            {
                Console.WriteLine("epoch " + (epoch + 1));
                RILogManager.Default.SendInformation("epoch " + (epoch + 1));
                RILogManager.Default.ViewerSendWatch("epoch", (epoch + 1));
                System.Windows.Forms.Application.DoEvents();

                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    System.Windows.Forms.Application.DoEvents();

                    TestDataSet datasetX = mnistData.GetRandomXSet(BATCH_DATA_COUNT,neuronCount, neuronCount);

                    Real sumLoss = Trainer.Train(nn, datasetX.Data, datasetX.Label, new SoftmaxCrossEntropy());
                    totalLoss += sumLoss;
                    totalLossCounter++;

                    stats.Push(sumLoss);
                    lossHistogram.AddBucket(new Bucket(-10, 10));
                    accuracyHistogram.AddBucket(new Bucket(-10.0, 10));

                    if (sumLoss < bestLocalLoss && sumLoss != Double.NaN)
                        bestLocalLoss = sumLoss;
                    if (stats.Mean < bestTotalLoss && sumLoss != Double.NaN)
                        bestTotalLoss = stats.Mean;

                    try
                    {
                        lossHistogram.AddData(sumLoss);
                    }
                    catch (Exception)
                    {
                    }

                    if (i % 20 == 0)
                    {
                        Console.WriteLine("\nbatch count " + i + "/" + TRAIN_DATA_COUNT + ", epoch " + epoch+1);
                        Console.WriteLine("Total/Mean loss " + stats.Mean);
                        Console.WriteLine("local loss " + sumLoss);
                        Console.WriteLine("");

                        RILogManager.Default.ViewerSendWatch("Batch Count", i);
                        RILogManager.Default.ViewerSendWatch("Total/Mean loss", stats.Mean);
                        RILogManager.Default.ViewerSendWatch("Local loss", sumLoss);
                        System.Windows.Forms.Application.DoEvents();


                        Console.WriteLine("Testing...");
                        TestDataSet datasetY = mnistData.GetRandomYSet(TEST_DATA_COUNT, 28);
                        Real accuracy = Trainer.Accuracy(nn, datasetY.Data, datasetY.Label);
                        if (accuracy > highestAccuracy)
                            highestAccuracy = accuracy;

                        Console.WriteLine("Accuracy: " + accuracy);


                        RILogManager.Default.ViewerSendWatch("Batch Count", i);
                        RILogManager.Default.ViewerSendWatch("Batch Count", i);
                        RILogManager.Default.ViewerSendWatch("Batch Count", i);
                        RILogManager.Default.ViewerSendWatch("Best Accuracy: ", highestAccuracy);
                        RILogManager.Default.ViewerSendWatch("Best Total Loss ", bestTotalLoss);
                        RILogManager.Default.ViewerSendWatch("Best Local Loss ", bestLocalLoss);
                        System.Windows.Forms.Application.DoEvents();

                        try
                        {
                            accuracyHistogram.AddData(accuracy);
                        }
                        catch (Exception)
                        {
                        }
                    }
                }
            }

            Console.WriteLine("Best Accuracy: " + highestAccuracy);
            Console.WriteLine("Best Total Loss " + bestTotalLoss);
            Console.WriteLine("Best Local Loss " + bestLocalLoss);
        }
    }
}

