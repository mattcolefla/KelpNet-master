using System;
using System.Diagnostics;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Noise;
using KelpNet.Functions.Poolings;

namespace KelpNetTester.Benchmarker
{
    using Nerdle.Ensure;
    using ReflectSoftware.Insight;

    class SingleBenchmark
    {
        // Assume the maximum memory of Linear of VGG 16
        const int INPUT_SIZE = 25088;
        const int OUTPUT_SIZE = 4096;


        private static void HandleGPU(bool verbose, Stopwatch sw, CompressibleActivation ca, NdArray nArray)
        {
            sw.Restart();
            NdArray[] gradArrayGpu = ca.Forward(verbose, nArray);
            sw.Stop();
            if (verbose)
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

            sw.Restart();
            ca.Backward(verbose, gradArrayGpu);
            sw.Stop();
            if (verbose)
                RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
        }
        private static void HandleGPU(bool verbose, Stopwatch sw, CompressibleFunction ca, NdArray nArray)
        {
            sw.Restart();
            NdArray[] gradArrayGpu = ca.Forward(verbose, nArray);
            sw.Stop();
            if (verbose)
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

            sw.Restart();
            ca.Backward(verbose, gradArrayGpu);
            sw.Stop();
            if (verbose)
                RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
        }

        private static void PrintResults(Stopwatch sw)
        {
            RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
        }

        public static void Run(bool verbose)
        {
            Stopwatch sw = new Stopwatch();

            NdArray inputArrayCpu = new NdArray(BenchDataMaker.GetRealArray(INPUT_SIZE));
            NdArray inputArrayGpu = new NdArray(BenchDataMaker.GetRealArray(INPUT_SIZE));
            Ensure.Argument(inputArrayGpu).NotNull();
            Ensure.Argument(inputArrayCpu).NotNull();

            //Linear
            Linear linear = new Linear(verbose, INPUT_SIZE, OUTPUT_SIZE);
            if (verbose)
                RILogManager.Default?.EnterMethod(linear.Name);

            sw.Restart();
            NdArray[] gradArrayCpu = linear.Forward(verbose, inputArrayCpu);
            sw.Stop();
            if (verbose)
                RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            Ensure.Argument(gradArrayCpu).NotNull();

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data; // Use Data as Grad

            sw.Restart();
            linear.Backward(verbose, gradArrayCpu);
            sw.Stop();
            if (verbose)
                RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (linear.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = linear.Forward(verbose, inputArrayGpu);
                sw.Stop();
                if (verbose)
                    RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                linear.Backward(verbose, gradArrayGpu);
                sw.Stop();
                if (verbose)
                    RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }
            if (verbose)
                RILogManager.Default?.ExitMethod(linear.Name);

            //Tanh
            Tanh tanh = new Tanh();
            if (verbose)
                RILogManager.Default?.EnterMethod(tanh.Name);

            sw.Restart();
            gradArrayCpu = tanh.Forward(verbose, inputArrayCpu);
            sw.Stop();
            if (verbose)
                RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            tanh.Backward(verbose, gradArrayCpu);
            sw.Stop();
            if (verbose)
                RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (tanh.SetGpuEnable(true))
                HandleGPU(verbose, sw, tanh, inputArrayGpu);

            if (verbose)
                RILogManager.Default?.ExitMethod(tanh.Name);



            //Sigmoid
            Sigmoid sigmoid = new Sigmoid();
            if (verbose)
                RILogManager.Default?.EnterMethod(sigmoid.Name);

            sw.Restart();
            gradArrayCpu = sigmoid.Forward(verbose, inputArrayCpu);
            sw.Stop();
            if (verbose)
                RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            sigmoid.Backward(verbose, gradArrayCpu);
            sw.Stop();
            if (verbose)
                RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (sigmoid.SetGpuEnable(true))
                HandleGPU(verbose, sw, sigmoid, inputArrayGpu);
            if (verbose)
                RILogManager.Default?.ExitMethod(tanh.Name);


            //Softmax
            Softmax sm = new Softmax();
            RILogManager.Default?.EnterMethod(sm.Name);

            sw.Restart();
            gradArrayCpu = sm.Forward(verbose, inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            sm.Backward(verbose, gradArrayCpu);
            sw.Stop();
            if (verbose)
              RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            if (verbose)
                RILogManager.Default?.ExitMethod(sm.Name);



            //Softplus
            Softplus sp = new Softplus();
            if (verbose)
                RILogManager.Default?.EnterMethod(sp.Name);

            sw.Restart();
            gradArrayCpu = sp.Forward(verbose, inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            sp.Backward(verbose, gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            RILogManager.Default?.ExitMethod(sp.Name);


            //ReLU
            ReLU relu = new ReLU();
            RILogManager.Default?.EnterMethod(relu.Name);

            sw.Restart();
            gradArrayCpu = relu.Forward(verbose, inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            relu.Backward(verbose, gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (relu.SetGpuEnable(true))
                HandleGPU(verbose, sw, relu, inputArrayGpu);

            RILogManager.Default?.ExitMethod(relu.Name);


            //LeakyReLU
            LeakyReLU leakyRelu = new LeakyReLU();
            RILogManager.Default?.EnterMethod(leakyRelu.Name);

            sw.Restart();
            gradArrayCpu = leakyRelu.Forward(verbose, inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            leakyRelu.Backward(verbose, gradArrayCpu);
            sw.Stop();

            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (leakyRelu.SetGpuEnable(true))
                HandleGPU(verbose, sw, leakyRelu, inputArrayGpu);
            RILogManager.Default?.ExitMethod(leakyRelu.Name);


            //ReLuTanh
            ReLuTanh rth = new ReLuTanh();
            RILogManager.Default?.EnterMethod(rth.Name);

            sw.Restart();
            gradArrayCpu = rth.Forward(verbose, inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            rth.Backward(verbose, gradArrayCpu);
            sw.Stop();

            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (rth.SetGpuEnable(true))
                HandleGPU(verbose, sw, rth, inputArrayGpu);
            RILogManager.Default?.ExitMethod(rth.Name);


            ////Swish
            //Swish swi = new Swish();
            //RILogManager.Default?.SendDebug(swi.Name);

            //sw.Restart();
            //gradArrayCpu = swi.Forward(inputArrayCpu);
            //sw.Stop();
            //RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            //gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            //sw.Restart();
            //swi.Backward(gradArrayCpu);
            //sw.Stop();

            //RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");


            NdArray inputImageArrayGpu = new NdArray(BenchDataMaker.GetRealArray(3 * 256 * 256 * 5), new[] { 3, 256, 256 }, 5);
            NdArray inputImageArrayCpu = new NdArray(BenchDataMaker.GetRealArray(3 * 256 * 256 * 5), new[] { 3, 256, 256 }, 5);

            //MaxPooling
            MaxPooling maxPooling = new MaxPooling(3);
            RILogManager.Default?.EnterMethod(maxPooling.Name);

            sw.Restart();
            NdArray[] gradImageArrayCpu = maxPooling.Forward(verbose, inputImageArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradImageArrayCpu[0].Grad = gradImageArrayCpu[0].Data;

            sw.Restart();
            maxPooling.Backward(verbose, gradImageArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (maxPooling.SetGpuEnable(true))
            {
                sw.Restart();
                maxPooling.Forward(verbose, inputImageArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                // There is no implementation for memory transfer only
                RILogManager.Default?.SendDebug("Backward[Gpu] : None");
            }
            RILogManager.Default?.ExitMethod(maxPooling.Name);


            //AvgPooling
            AveragePooling avgPooling = new AveragePooling(3);
            RILogManager.Default?.EnterMethod(avgPooling.Name);

            sw.Restart();
            gradImageArrayCpu = avgPooling.Forward(verbose, inputImageArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradImageArrayCpu[0].Grad = gradImageArrayCpu[0].Data;

            sw.Restart();
            avgPooling.Backward(verbose, gradImageArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            RILogManager.Default?.ExitMethod(avgPooling.Name);


            //Conv2D
            Convolution2D conv2d = new Convolution2D(verbose, 3, 3, 3);
            RILogManager.Default?.EnterMethod(conv2d.Name);

            sw.Restart();
            gradImageArrayCpu = conv2d.Forward(verbose, inputImageArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradImageArrayCpu[0].Grad = gradImageArrayCpu[0].Data;

            sw.Restart();
            conv2d.Backward(verbose, gradImageArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (conv2d.SetGpuEnable(true))
                HandleGPU(verbose, sw, conv2d, inputArrayGpu);

            RILogManager.Default?.ExitMethod(conv2d.Name);


            //Deconv2D
            Deconvolution2D deconv2d = new Deconvolution2D(verbose, 3, 3, 3);
            RILogManager.Default?.EnterMethod(deconv2d.Name);

            sw.Restart();
            gradImageArrayCpu = deconv2d.Forward(verbose, inputImageArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradImageArrayCpu[0].Grad = gradImageArrayCpu[0].Data;

            sw.Restart();
            deconv2d.Backward(verbose, gradImageArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (deconv2d.SetGpuEnable(true))
                HandleGPU(verbose, sw, deconv2d, inputArrayGpu);

            RILogManager.Default?.ExitMethod(deconv2d.Name);


            //Dropout
            Dropout dropout = new Dropout();
            RILogManager.Default?.EnterMethod(dropout.Name);

            sw.Restart();
            gradArrayCpu = dropout.Forward(verbose, inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            dropout.Backward(verbose, gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (dropout.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = dropout.Forward(verbose, inputArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                dropout.Backward(verbose, gradArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }
            RILogManager.Default?.ExitMethod(dropout.Name);

            //ArcSinH
            ArcSinH a = new ArcSinH();
            RILogManager.Default?.EnterMethod(a.Name);

            sw.Restart();
            gradArrayCpu = a.Forward(verbose, inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            a.Backward(verbose, gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (a.SetGpuEnable(true))
                HandleGPU(verbose, sw, a, inputArrayGpu);
            RILogManager.Default?.ExitMethod(a.Name);

            //ELU
            ELU e = new ELU();
            RILogManager.Default?.EnterMethod(e.Name);

            sw.Restart();
            gradArrayCpu = e.Forward(verbose, inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            e.Backward(verbose, gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            RILogManager.Default?.ExitMethod(e.Name);

            //LeakyReluShifted
            LeakyReLUShifted lrs = new LeakyReLUShifted();
            RILogManager.Default?.EnterMethod(lrs.Name);

            sw.Restart();
            gradArrayCpu = lrs.Forward(verbose, inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            lrs.Backward(verbose, gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (lrs.SetGpuEnable(true))
                HandleGPU(verbose, sw, lrs, inputArrayGpu);
            RILogManager.Default?.ExitMethod(lrs.Name);


            //Logistic
            LogisticFunction lf = new LogisticFunction();
            RILogManager.Default?.EnterMethod(lf.Name);

            sw.Restart();
            gradArrayCpu = lf.Forward(verbose, inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            lf.Backward(verbose, gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (lf.SetGpuEnable(true))
                HandleGPU(verbose, sw, lf, inputArrayGpu);
            RILogManager.Default?.ExitMethod(lf.Name);


            //MaxMinusOne
            MaxMinusOne mmo = new MaxMinusOne();
            RILogManager.Default?.EnterMethod(mmo.Name);

            sw.Restart();
            gradArrayCpu = mmo.Forward(verbose, inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            mmo.Backward(verbose, gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (mmo.SetGpuEnable(true))
                HandleGPU(verbose, sw, mmo, inputArrayGpu);
            RILogManager.Default?.ExitMethod(mmo.Name);


            //ScaledELU
            ScaledELU se = new ScaledELU();
            RILogManager.Default?.EnterMethod(se.Name);

            sw.Restart();
            gradArrayCpu = se.Forward(verbose, inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            se.Backward(verbose, gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (se.SetGpuEnable(true))
                HandleGPU(verbose, sw, se, inputArrayGpu);
            RILogManager.Default?.ExitMethod(se.Name);


            //Sine
            Sine s = new Sine();
            RILogManager.Default?.EnterMethod(s.Name);

            sw.Restart();
            gradArrayCpu = s.Forward(verbose, inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            s.Backward(verbose, gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (s.SetGpuEnable(true))
                HandleGPU(verbose, sw, s, inputArrayGpu);
            RILogManager.Default?.ExitMethod(s.Name);



        }
    }
}
