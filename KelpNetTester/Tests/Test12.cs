using System;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions;
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

    //Decoupled Neural Interfaces using Synthetic Gradients Learning MNIST (Handwritten Characters) by Gradients
    // Incorporate label information into teacher signal cDNI
    // Execute Decoupled of all layers expressed as Model D asynchronously
    // http://ralo23.hatenablog.com/entry/2016/10/22/233405
    class Test12
    {
        // number of mini-batches
        const int BATCH_DATA_COUNT = 256;

        // Number of exercises per generation
        const int TRAIN_DATA_COUNT = 234; // = 60,000 / 256

        // number of data at performance evaluation
        const int TEST_DATA_COUNT = 100;

        class ResultDataSet
        {
            public readonly NdArray[] Result;
            public readonly NdArray Label;

            public ResultDataSet(NdArray[] result, NdArray label)
            {
                Result = result;
                Label = label;
            }

            public NdArray GetTrainData()
            {
                Real[] tmp = new Real[(256 + 10) * BATCH_DATA_COUNT];

                for (int i = 0; i < BATCH_DATA_COUNT; i++)
                {
                    tmp[256 + (int)Label.Data[i * Label.Length] + i * (256 + 10)] = 1;
                    Array.Copy(Result[0].Data, i * Result.Length, tmp, i * (256 + 10), 256);
                }

                return NdArray.Convert(tmp, new[] { 256 + 10 }, BATCH_DATA_COUNT, (Function)null);
            }
        }

        public static void Run()
        {
            // Prepare MNIST data
            RILogManager.Default?.SendDebug("MNIST Data Loading...");
            MnistData mnistData = new MnistData(28);

            RILogManager.Default?.SendDebug("Training Start...");

            // Write the network configuration in FunctionStack
            FunctionStack Layer1 = new FunctionStack("Test12 Layer 1",
                new Linear(true, 28 * 28, 256, name: "l1 Linear"),
                new BatchNormalization(true, 256, name: "l1 Norm"),
                new ReLU(name: "l1 ReLU")
            );

            FunctionStack Layer2 = new FunctionStack("Test12 Layer 2",
                new Linear(true, 256, 256, name: "l2 Linear"),
                new BatchNormalization(true, 256, name: "l2 Norm"),
                new ReLU(name: "l2 ReLU")
            );

            FunctionStack Layer3 = new FunctionStack("Test12 Layer 3",
                new Linear(true, 256, 256, name: "l3 Linear"),
                new BatchNormalization(true, 256, name: "l3 Norm"),
                new ReLU(name: "l3 ReLU")
            );

            FunctionStack Layer4 = new FunctionStack("Test12 Layer 4",
                new Linear(true, 256, 10, name: "l4 Linear")
            );

            // Function stack itself is also stacked as Function
            FunctionStack nn = new FunctionStack
            ("Test12",
                Layer1,
                Layer2,
                Layer3,
                Layer4
            );

            FunctionStack cDNI1 = new FunctionStack("Test12 DNI 1",
                new Linear(true, 256 + 10, 1024, name: "cDNI1 Linear1"),
                new BatchNormalization(true, 1024, name: "cDNI1 Norm1"),
                new ReLU(name: "cDNI1 ReLU1"),
                new Linear(true, 1024, 256, initialW: new Real[1024, 256], name: "DNI1 Linear3")
            );

            FunctionStack cDNI2 = new FunctionStack("Test12 DNI 2",
                new Linear(true, 256 + 10, 1024, name: "cDNI2 Linear1"),
                new BatchNormalization(true, 1024, name: "cDNI2 Norm1"),
                new ReLU(name: "cDNI2 ReLU1"),
                new Linear(true, 1024, 256, initialW: new Real[1024, 256], name: "cDNI2 Linear3")
            );

            FunctionStack cDNI3 = new FunctionStack("Test12 DNI 3",
                new Linear(true, 256 + 10, 1024, name: "cDNI3 Linear1"),
                new BatchNormalization(true, 1024, name: "cDNI3 Norm1"),
                new ReLU(name: "cDNI3 ReLU1"),
                new Linear(true, 1024, 256, initialW: new Real[1024, 256], name: "cDNI3 Linear3")
            );

            Layer1.SetOptimizer(new Adam("Adam",0.00003f));
            Layer2.SetOptimizer(new Adam("Adam", 0.00003f));
            Layer3.SetOptimizer(new Adam("Adam", 0.00003f));
            Layer4.SetOptimizer(new Adam("Adam", 0.00003f));

            cDNI1.SetOptimizer(new Adam("Adam", 0.00003f));
            cDNI2.SetOptimizer(new Adam("Adam", 0.00003f));
            cDNI3.SetOptimizer(new Adam("Adam", 0.00003f));

            // Describe each function stack;
            RILogManager.Default?.SendDebug(Layer1.Describe());
            RILogManager.Default?.SendDebug(Layer2.Describe());
            RILogManager.Default?.SendDebug(Layer3.Describe());
            RILogManager.Default?.SendDebug(Layer4.Describe());

            RILogManager.Default?.SendDebug(cDNI1.Describe());
            RILogManager.Default?.SendDebug(cDNI2.Describe());
            RILogManager.Default?.SendDebug(cDNI3.Describe());

            for (int epoch = 0; epoch < 10; epoch++)
            {
                // Total error in the whole
                Real totalLoss = 0;
                Real cDNI1totalLoss = 0;
                Real cDNI2totalLoss = 0;
                Real cDNI3totalLoss = 0;

                long totalLossCount = 0;
                long cDNI1totalLossCount = 0;
                long cDNI2totalLossCount = 0;
                long cDNI3totalLossCount = 0;


                // how many times to run the batch
                for (int i = 1; i < TRAIN_DATA_COUNT + 1; i++)
                {
                    RILogManager.Default?.SendDebug("epoch: " + (epoch + 1) + " of 10, batch iteration: " + i + " of " + TRAIN_DATA_COUNT);
                    RILogManager.Default?.ViewerSendWatch("Epoch", epoch + 1);
                    RILogManager.Default?.ViewerSendWatch("Batch Iteration", i);

                    // Get data randomly from the training data
                    TestDataSet datasetX = mnistData.GetRandomXSet(BATCH_DATA_COUNT,28,28);

                    // Run first tier
                    NdArray[] layer1ForwardResult = Layer1.Forward(true, datasetX.Data);
                    ResultDataSet layer1ResultDataSet = new ResultDataSet(layer1ForwardResult, datasetX.Label);

                    // Obtain the slope of the first layer
                    NdArray[] cDNI1Result = cDNI1.Forward(true, layer1ResultDataSet.GetTrainData());

                    // Apply the slope of the first layer
                    layer1ForwardResult[0].Grad = cDNI1Result[0].Data.ToArray();

                    //Update first layer
                    Layer1.Backward(true, layer1ForwardResult);
                    layer1ForwardResult[0].ParentFunc = null;
                    Layer1.Update();

                    // Run Layer 2
                    NdArray[] layer2ForwardResult = Layer2.Forward(true, layer1ResultDataSet.Result);
                    ResultDataSet layer2ResultDataSet = new ResultDataSet(layer2ForwardResult, layer1ResultDataSet.Label);

                    // Get the inclination of the second layer
                    NdArray[] cDNI2Result = cDNI2.Forward(true, layer2ResultDataSet.GetTrainData());

                    // Apply the slope of the second layer
                    layer2ForwardResult[0].Grad = cDNI2Result[0].Data.ToArray();

                    //Update layer 2
                    Layer2.Backward(true, layer2ForwardResult);
                    layer2ForwardResult[0].ParentFunc = null;


                    //Perform learning of first layer cDNI
                    Real cDNI1loss = new MeanSquaredError().Evaluate(cDNI1Result, new NdArray(layer1ResultDataSet.Result[0].Grad, cDNI1Result[0].Shape, cDNI1Result[0].BatchCount));

                    Layer2.Update();

                    cDNI1.Backward(true, cDNI1Result);
                    cDNI1.Update();

                    cDNI1totalLoss += cDNI1loss;
                    cDNI1totalLossCount++;

                    //Run Third Tier
                    NdArray[] layer3ForwardResult = Layer3.Forward(true, layer2ResultDataSet.Result);
                    ResultDataSet layer3ResultDataSet = new ResultDataSet(layer3ForwardResult, layer2ResultDataSet.Label);

                    //Get the inclination of the third layer
                    NdArray[] cDNI3Result = cDNI3.Forward(true, layer3ResultDataSet.GetTrainData());

                    //Apply the inclination of the third layer
                    layer3ForwardResult[0].Grad = cDNI3Result[0].Data.ToArray();

                    //Update third layer
                    Layer3.Backward(true, layer3ForwardResult);
                    layer3ForwardResult[0].ParentFunc = null;

                    //Perform learning of cDNI for layer 2
                    Real cDNI2loss = new MeanSquaredError().Evaluate(cDNI2Result, new NdArray(layer2ResultDataSet.Result[0].Grad, cDNI2Result[0].Shape, cDNI2Result[0].BatchCount));

                    Layer3.Update();

                    cDNI2.Backward(true, cDNI2Result);
                    cDNI2.Update();

                    cDNI2totalLoss += cDNI2loss;
                    cDNI2totalLossCount++;

                    NdArray[] layer4ForwardResult = Layer4.Forward(true, layer3ResultDataSet.Result);
                    Real sumLoss = new SoftmaxCrossEntropy().Evaluate(layer4ForwardResult, layer3ResultDataSet.Label);
                    Layer4.Backward(true, layer4ForwardResult);
                    layer4ForwardResult[0].ParentFunc = null;

                    totalLoss += sumLoss;
                    totalLossCount++;

                    Real cDNI3loss = new MeanSquaredError().Evaluate(cDNI3Result, new NdArray(layer3ResultDataSet.Result[0].Grad, cDNI3Result[0].Shape, cDNI3Result[0].BatchCount));

                    Layer4.Update();

                    cDNI3.Backward(true, cDNI3Result);
                    cDNI3.Update();

                    cDNI3totalLoss += cDNI3loss;
                    cDNI3totalLossCount++;

                    RILogManager.Default?.SendDebug("\nbatch count " + i + "/" + TRAIN_DATA_COUNT);
                    RILogManager.Default?.SendDebug("total loss " + totalLoss / totalLossCount);
                    RILogManager.Default?.SendDebug("local loss " + sumLoss);

                    RILogManager.Default?.SendDebug("\ncDNI1 total loss " + cDNI1totalLoss / cDNI1totalLossCount);
                    RILogManager.Default?.SendDebug("cDNI2 total loss " + cDNI2totalLoss / cDNI2totalLossCount);
                    RILogManager.Default?.SendDebug("cDNI3 total loss " + cDNI3totalLoss / cDNI3totalLossCount);

                    RILogManager.Default?.SendDebug("\ncDNI1 local loss " + cDNI1loss);
                    RILogManager.Default?.SendDebug("cDNI2 local loss " + cDNI2loss);
                    RILogManager.Default?.SendDebug("cDNI3 local loss " + cDNI3loss);

                    if (i % 20 == 0)
                    {
                        RILogManager.Default?.SendDebug("\nTesting...");
                        TestDataSet datasetY = mnistData.GetRandomYSet(TEST_DATA_COUNT, 28);
                        Real accuracy = Trainer.Accuracy(nn, datasetY.Data, datasetY.Label);
                        RILogManager.Default?.SendDebug("accuracy " + accuracy);
                    }
                }
            }
        }
    }
}
