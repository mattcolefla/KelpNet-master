using System;
using KelpNet.Common;
using KelpNetTester.Benchmarker;
using KelpNetTester.Tests;

namespace KelpNetTester
{
    using System.Diagnostics;
    using System.IO;
    using System.Threading;
    using HdrHistogram;
    using ReflectSoftware.Insight;

    //Please uncomment the test you want to run
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            //Comment out here if you want to run all on .Net Framework
            Weaver.Initialize(ComputeDeviceTypes.Gpu);
            //Weaver.Initialize(ComputeDeviceTypes.Cpu, 1); //Subscript required if there are multiple devices

            //Learning XOR with MLP
            //recorder.Record(() => Test1.Run());

            //Learning XOR with MLP 【Returned version】
            //Test2.Run();

            //Learning of Sin function by MLP
            //recorder.Record(() => Test3.Run());

            //Learning of MNIST (Handwritten Characters) by MLP
            //Test4.Run();

            //Reproduction of Excel CNN
            //recorder.Record(() => Test5.Run());

            //Learning of MNIST by 5-layer CNN
            //recorder.Record(() => Test6.Run());

            //Learning of MNIST by 15-tier MLP using BatchNorm
            //Test7.Run();

            //Learning of Sin function by LSTM
            //Test8.Run();

            //RNNLM with Simple RNN
            //Test9.Run();

            //RNNLM by LSTM
            //Test10.Run();

            //Decoupled Neural Interfaces using Synthetic Gradients by learning MNIST
            //Test11.Run();

            //DNI of Test 11 was defined as cDNI
            //Test12.Run();

            //Test of Deconvolution 2D(Winform)
            //new Test13WinForm().ShowDialog();

            //Concatenate Test 6 and execute
            //Test14.Run();

            //Test to read VGA 16 of Caffe model and classify images
            //Test15.Run();

            //Load and execute the same content as Test 5 of Chainer model
            //Test16.Run();

            //Test that reads RESNET of Caffe model and classifies images
            //Test17.Run(Test17.ResnetModel.ResNet50);  //Please select any Resnet model

            //Learn CIFAR-10 with 5-layer CNN
            //Test18.Run();

            //Partial execution of Linear
            //TestX.Run();

            // LeakyReLu and PolynomialApproximantSteep combination network
            //Test19.Run();

            // 1000 layer neural network
            Test20.Run();

            //benchmark
            //SingleBenchmark.Run();

            RILogManager.Default?.SendDebug("Test Done...");
            Console.WriteLine("Test Complete, press any key to end");
            Console.Read();
        }
    }
}
