using System;
using KelpNet.Common;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Poolings;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    using ReflectSoftware.Insight;

    // Reproduction of Excel CNN
    class Test5
    {
        public static void Run()
        {
            // Describe each initial value
            Real[,,,] initial_W1 =
                {
                    {{{1.0,  0.5, 0.0}, { 0.5, 0.0, -0.5}, {0.0, -0.5, -1.0}}},
                    {{{0.0, -0.1, 0.1}, {-0.3, 0.4,  0.7}, {0.5, -0.2,  0.2}}}
                };
            Real[] initial_b1 = { 0.5, 1.0 };

            Real[,,,] initial_W2 =
                {
                    {{{-0.1,  0.6}, {0.3, -0.9}}, {{ 0.7, 0.9}, {-0.2, -0.3}}},
                    {{{-0.6, -0.1}, {0.3,  0.3}}, {{-0.5, 0.8}, { 0.9,  0.1}}}
                };
            Real[] initial_b2 = { 0.1, 0.9 };

            Real[,] initial_W3 =
                {
                    {0.5, 0.3, 0.4, 0.2, 0.6, 0.1, 0.4, 0.3},
                    {0.6, 0.4, 0.9, 0.1, 0.5, 0.2, 0.3, 0.4}
                };
            Real[] initial_b3 = { 0.01, 0.02 };

            Real[,] initial_W4 = { { 0.8, 0.2 }, { 0.4, 0.6 } };
            Real[] initial_b4 = { 0.02, 0.01 };


            //Input data
            NdArray x = new NdArray(new Real[, ,]{{
                    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.9, 0.2, 0.0, 0.0, 0.0, 0.0},
                    { 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0},
                    { 0.0, 0.0, 0.0, 0.1, 0.8, 0.5, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0},
                    { 0.0, 0.0, 0.0, 0.3, 0.3, 0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0},
                    { 0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0},
                    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.1, 0.0, 0.0, 0.0, 0.0},
                    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0},
                    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0},
                    { 0.0, 0.0, 0.0, 0.0, 0.2, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0},
                    { 0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0},
                    { 0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0},
                    { 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
                }});

            //teacher signal
            Real[] t = { 0.0, 1.0 };


            // If you want to check the contents of a layer, have an instance as a single layer
            Convolution2D l2 = new Convolution2D(1, 2, 3, initialW: initial_W1, initialb: initial_b1, name: "l2 Conv2D");

            // Write the network configuration in FunctionStack
            FunctionStack nn = new FunctionStack(
                l2, //new Convolution2D(1, 2, 3, initialW: initial_W1, initialb: initial_b1),
                new ReLU(name: "l2 ReLU"),
                //new AveragePooling(2, 2, name: "l2 AVGPooling"),
                new MaxPooling(2, 2, name: "l2 MaxPooling"),
                new Convolution2D(2, 2, 2, initialW: initial_W2, initialb: initial_b2, name: "l3 Conv2D"),
                new ReLU(name: "l3 ReLU"),
                //new AveragePooling(2, 2, name: "l3 AVGPooling"),
                new MaxPooling(2, 2, name: "l3 MaxPooling"),
                new Linear(8, 2, initialW: initial_W3, initialb: initial_b3, name: "l4 Linear"),
                new ReLU(name: "l4 ReLU"),
                new Linear(2, 2, initialW: initial_W4, initialb: initial_b4, name: "l5 Linear")
            );

            // If you omit the optimizer declaration, the default SGD(0.1) will be used nn.SetOptimizer(new SGD());

            // Training conducted            Trainer.Train(nn, x, t, new MeanSquaredError(), false);

            // If Update is executed, grad is consumed, so output the value first
            RILogManager.Default?.SendDebug("gw1");
            RILogManager.Default?.SendDebug(l2.Weight.ToString("Grad"));

            RILogManager.Default?.SendDebug("gb1");
            RILogManager.Default?.SendDebug(l2.Bias.ToString("Grad"));

            //update
            nn.Update();

            RILogManager.Default?.SendDebug("w1");
            RILogManager.Default?.SendDebug(l2.Weight.ToString());

            RILogManager.Default?.SendDebug("b1");
            RILogManager.Default?.SendDebug(l2.Bias.ToString());
        }
    }
}
