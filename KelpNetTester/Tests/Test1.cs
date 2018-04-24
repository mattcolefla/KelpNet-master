using System;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    using System.Diagnostics.CodeAnalysis;
    using ReflectSoftware.Insight;

    //Learning XOR
    [SuppressMessage("ReSharper", "LocalizableElement")]
    [SuppressMessage("ReSharper", "AssignNullToNotNullAttribute")]
    [SuppressMessage("ReSharper", "PossibleNullReferenceException")]
    class Test1
    {
        public static void Run()
        {
            const int learningCount = 10000;

            Real[][] trainData =
            {
                new Real[] { 0, 0 },
                new Real[] { 1, 0 },
                new Real[] { 0, 1 },
                new Real[] { 1, 1 }
            };

            Real[][] trainLabel =
            {
                new Real[] { 0 },
                new Real[] { 1 },
                new Real[] { 1 },
                new Real[] { 0 }
            };

            FunctionStack nn = new FunctionStack(
                new Linear(2, 2, name: "l1 Linear"),
                new Sigmoid(name: "l1 Sigmoid"),
                new Linear(2, 2, name: "l2 Linear"));

            nn.SetOptimizer(new MomentumSGD());

            RILogManager.Default?.SendDebug("Training...");
            for (int i = 0; i < learningCount; i++)
            {
                for (int j = 0; j < trainData.Length; j++)
                {
                    Trainer.Train(nn, trainData[j], trainLabel[j], new SoftmaxCrossEntropy());
                }
            }

            RILogManager.Default?.SendDebug("Test Start...");

            foreach (Real[] input in trainData)
            {
                NdArray result = nn.Predict(input)?[0];
                int resultIndex = Array.IndexOf(result?.Data, result.Data.Max());
                RILogManager.Default?.SendDebug($"{input[0]} xor {input[1]} = {resultIndex} {result}");
            }

            ModelIO.Save(nn, "test.nn");

            FunctionStack testnn = ModelIO.Load("test.nn");
            RILogManager.Default?.SendDebug("Test Start...");
            foreach (Real[] input in trainData)
            {
                NdArray result = testnn?.Predict(input)?[0];
                int resultIndex = Array.IndexOf(result?.Data, result?.Data.Max());
                RILogManager.Default?.SendDebug($"{input[0]} xor {input[1]} = {resultIndex} {result}");
            }
        }
    }
}
