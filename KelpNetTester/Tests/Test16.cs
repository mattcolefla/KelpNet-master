using System;
using ChainerModelLoader;
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

    class Test16
    {
        private const string MODEL_FILE_PATH = "Data/ChainerModel.npz";

        public static void Run()
        {
            // Write the configuration of the network you want to read into FunctionStack and adjust the parameters of each function
            // Make sure to match name to the variable name of Chainer here

            FunctionStack nn = new FunctionStack(
                new Convolution2D(1, 2, 3, name: "conv1", gpuEnable: true),// Do not forget the GPU flag if necessary
                new ReLU(),
                new MaxPooling(2, 2),
                new Convolution2D(2, 2, 2, name: "conv2", gpuEnable: true),
                new ReLU(),
                new MaxPooling(2, 2),
                new Linear(8, 2, name: "fl3"),
                new ReLU(),
                new Linear(2, 2, name: "fl4")
            );

            /* Chainerでの宣言
            class NN(chainer.Chain):
                def __init__(self):
                    super(NN, self).__init__(
                        conv1 = L.Convolution2D(1,2,3),
                        conv2 = L.Convolution2D(2,2,2),
                        fl3 = L.Linear(8,2),
                        fl4 = L.Linear(2,2)
                    )

                def __call__(self, x):
                    h_conv1 = F.relu(self.conv1(x))
                    h_pool1 = F.max_pooling_2d(h_conv1, 2)
                    h_conv2 = F.relu(self.conv2(h_pool1))
                    h_pool2 = F.max_pooling_2d(h_conv2, 2)
                    h_fc1 = F.relu(self.fl3(h_pool2))
                    y = self.fl4(h_fc1)
                    return y
            */


            // Read parameters
            ChainerModelDataLoader.ModelLoad(MODEL_FILE_PATH, nn);

            // We will use the rest as usual
            nn.SetOptimizer(new SGD());

            NdArray x = new NdArray(new Real[,,]{{
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

            Real[] t = { 0.0, 1.0 };

            Trainer.Train(nn, x, t, new MeanSquaredError(), false);

            Convolution2D l2 = (Convolution2D)nn.Functions[0];


            RILogManager.Default?.SendDebug("gw1");
            RILogManager.Default?.SendDebug(l2.Weight.ToString("Grad"));
            RILogManager.Default?.SendDebug("gb1");
            RILogManager.Default?.SendDebug(l2.Bias.ToString("Grad"));


            // If Update is executed, grad is consumed, so output the value first
            nn.Update();

            RILogManager.Default?.SendDebug("w1");
            RILogManager.Default?.SendDebug(l2.Weight.ToString());

            RILogManager.Default?.SendDebug("b1");
            RILogManager.Default?.SendDebug(l2.Bias.ToString());
        }
    }
}
