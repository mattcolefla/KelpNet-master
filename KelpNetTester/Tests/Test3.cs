using System;
using KelpNet.Common;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    using ReflectSoftware.Insight;

    //Learning of Sin function by MLP

    //Increasing the period of the learning object or increasing the sampling number (N) will degrade the score,
    class Test3
    {
        const int EPOCH = 1000;

        //Number of divisions per period
        const int N = 50;

        public static void Run()
        {
            Real[][] trainData = new Real[N][];
            Real[][] trainLabel = new Real[N][];

            for (int i = 0; i < N; i++)
            {
                //Prepare Sin wave for one cycle
                Real radian = -Math.PI + Math.PI * 2.0 * i / (N - 1);
                trainData[i] = new[] { radian };
                trainLabel[i] = new Real[] { Math.Sin(radian) };
            }

            FunctionStack nn = new FunctionStack(
                new Linear(1, 4, name: "l1 Linear"),
                new Tanh(name: "l1 Tanh"),
                new Linear(4, 1, name: "l2 Linear")
            );

            nn.SetOptimizer(new SGD());

            for (int i = 0; i < EPOCH; i++)
            {
                Real loss = 0;

                for (int j = 0; j < N; j++)
                {
                    //When training is executed in the network, an error is returned to the return value
                    loss += Trainer.Train(nn, trainData[j], trainLabel[j], new MeanSquaredError());
                }

                if (i % (EPOCH / 10) == 0)
                {
                    RILogManager.Default?.SendDebug("loss:" + loss / N);
                    RILogManager.Default?.SendDebug("");
                }
            }

            RILogManager.Default?.SendDebug("Test Start...");
            foreach (Real[] val in trainData)
            {
                RILogManager.Default?.SendDebug(val[0] + ":" + nn.Predict(val)[0].Data[0]);
            }
        }
    }
}
