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
using System.Windows.Forms;

namespace KelpNetTester.Tests
{
    using System.Threading.Tasks;
    using KelpNet.Common.Functions;
    using ReflectSoftware.Insight;

    class Test20
    {
        // creating a massive deep learning neural network
        const int BATCH_DATA_COUNT = 128;
        private const int TRAIN_DATA_COUNT = 50000;
        const int TEST_DATA_COUNT = 200;
        private const int N = 30;
        private const int numLayers = 10000;

        public static void Run()
        {
            int neuronCount = 28;
            Console.WriteLine("MNIST Data Loading...");
            MnistData mnistData = new MnistData(neuronCount);
            RILogManager.Default.SendInformation("Training Start, creating function stack.");

            FunctionStack nn = new FunctionStack();
            List<Function> functions = new List<Function>();

            ParallelOptions po = new ParallelOptions();
            po.MaxDegreeOfParallelism = 4;

            for (int x=0; x< numLayers; x++)
            {
                Application.DoEvents();

                functions.Add(new Linear(neuronCount * neuronCount, N, name: $"l{x} Linear"));
                functions.Add(new BatchNormalization(N, name: $"l{x} BatchNorm"));
                functions.Add(new ReLU(name: $"l{x} ReLU"));
                RILogManager.Default.ViewerSendWatch("Total Layers", (x + 1));
            };

            RILogManager.Default.SendInformation("Adding Output Layer");
            Application.DoEvents();
            nn.Add(new Linear(N, 10, noBias: false, name: $"l{numLayers + 1} Linear"));
            RILogManager.Default.ViewerSendWatch("Total Layers", numLayers);


            RILogManager.Default.SendInformation("Setting Optimizer to AdaGrad");
            nn.SetOptimizer(new AdaGrad());
            Application.DoEvents();

            RunningStatistics stats = new RunningStatistics();
            Histogram lossHistogram = new Histogram();
            Histogram accuracyHistogram = new Histogram();
            Real totalLoss = 0;
            long totalLossCounter = 0;
            Real highestAccuracy = 0;
            Real bestLocalLoss = 0;
            Real bestTotalLoss = 0;

            for (int epoch = 0; epoch < 3; epoch++)
            {
                Console.WriteLine("epoch " + (epoch + 1));
                RILogManager.Default.SendInformation("epoch " + (epoch + 1));
                RILogManager.Default.ViewerSendWatch("epoch", (epoch + 1));
                Application.DoEvents();

                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    Application.DoEvents();

                    TestDataSet datasetX = mnistData.GetRandomXSet(BATCH_DATA_COUNT,neuronCount, neuronCount);

                    Real sumLoss = Trainer.Train(nn, datasetX.Data, datasetX.Label, new SoftmaxCrossEntropy());
                    totalLoss += sumLoss;
                    totalLossCounter++;

                    stats.Push(sumLoss);
                    lossHistogram.AddBucket(new Bucket(-10, 10));
                    accuracyHistogram.AddBucket(new Bucket(-10.0, 10));

                    if (sumLoss < bestLocalLoss && !double.IsNaN(sumLoss))
                        bestLocalLoss = sumLoss;
                    if (stats.Mean < bestTotalLoss && !double.IsNaN(sumLoss))
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

                        RILogManager.Default.ViewerSendWatch("Batch Count ", i);
                        RILogManager.Default.ViewerSendWatch("Total/Mean loss", stats.Mean);
                        RILogManager.Default.ViewerSendWatch("Local loss", sumLoss);
                        RILogManager.Default.SendInformation("Batch Count " + i + "/" + TRAIN_DATA_COUNT + ", epoch " + epoch + 1);
                        RILogManager.Default.SendInformation("Total/Mean loss " +  stats.Mean);
                        RILogManager.Default.SendInformation("Local loss " + sumLoss);
                        Application.DoEvents();


                        Console.WriteLine("Testing...");
                        RILogManager.Default.SendInformation("Testing");

                        TestDataSet datasetY = mnistData.GetRandomYSet(TEST_DATA_COUNT, 28);
                        Real accuracy = Trainer.Accuracy(nn, datasetY?.Data, datasetY.Label);
                        if (accuracy > highestAccuracy)
                            highestAccuracy = accuracy;

                        Console.WriteLine("Accuracy: " + accuracy);

                        RILogManager.Default.ViewerSendWatch("Best Accuracy: ", highestAccuracy);
                        RILogManager.Default.ViewerSendWatch("Best Total Loss ", bestTotalLoss);
                        RILogManager.Default.ViewerSendWatch("Best Local Loss ", bestLocalLoss);
                        Application.DoEvents();

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

            ModelIO.Save(nn, Application.StartupPath + "\\test20.nn");
            Console.WriteLine("Best Accuracy: " + highestAccuracy);
            Console.WriteLine("Best Total Loss " + bestTotalLoss);
            Console.WriteLine("Best Local Loss " + bestLocalLoss);
            RILogManager.Default.ViewerSendWatch("Best Accuracy: ", highestAccuracy);
            RILogManager.Default.ViewerSendWatch("Best Total Loss ", bestTotalLoss);
            RILogManager.Default.ViewerSendWatch("Best Local Loss ", bestLocalLoss);
        }
    }
}

