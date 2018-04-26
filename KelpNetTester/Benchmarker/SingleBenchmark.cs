using System;
using System.Diagnostics;
using KelpNet.Common;
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

        public static void Run()
        {
            Stopwatch sw = new Stopwatch();

            NdArray inputArrayCpu = new NdArray(BenchDataMaker.GetRealArray(INPUT_SIZE));
            NdArray inputArrayGpu = new NdArray(BenchDataMaker.GetRealArray(INPUT_SIZE));
            Ensure.Argument(inputArrayGpu).NotNull();
            Ensure.Argument(inputArrayCpu).NotNull();

            //Linear
            Linear linear = new Linear(INPUT_SIZE, OUTPUT_SIZE);
            RILogManager.Default?.EnterMethod(linear.Name);

            sw.Restart();
            NdArray[] gradArrayCpu = linear.Forward(inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data; // Use Data as Grad

            sw.Restart();
            linear.Backward(gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (linear.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = linear.Forward(inputArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                linear.Backward(gradArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }
            RILogManager.Default?.ExitMethod(linear.Name);

            //Tanh
            Tanh tanh = new Tanh();
            RILogManager.Default?.EnterMethod(tanh.Name);

            sw.Restart();
            gradArrayCpu = tanh.Forward(inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            tanh.Backward(gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (tanh.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = tanh.Forward(inputArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                tanh.Backward(gradArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }
            RILogManager.Default?.ExitMethod(tanh.Name);



            //Sigmoid
            Sigmoid sigmoid = new Sigmoid();
            RILogManager.Default?.EnterMethod(sigmoid.Name);

            sw.Restart();
            gradArrayCpu = sigmoid.Forward(inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            sigmoid.Backward(gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (sigmoid.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = sigmoid.Forward(inputArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                sigmoid.Backward(gradArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }
            RILogManager.Default?.ExitMethod(tanh.Name);


            //Softmax
            Softmax sm = new Softmax();
            RILogManager.Default?.EnterMethod(sm.Name);

            sw.Restart();
            gradArrayCpu = sm.Forward(inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            sm.Backward(gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            RILogManager.Default?.ExitMethod(sm.Name);



            //Softplus
            Softplus sp = new Softplus();
            RILogManager.Default?.EnterMethod(sp.Name);

            sw.Restart();
            gradArrayCpu = sp.Forward(inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            sp.Backward(gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            RILogManager.Default?.ExitMethod(sp.Name);


            //ReLU
            ReLU relu = new ReLU();
            RILogManager.Default?.EnterMethod(relu.Name);

            sw.Restart();
            gradArrayCpu = relu.Forward(inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            relu.Backward(gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (relu.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = relu.Forward(inputArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                relu.Backward(gradArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }
            RILogManager.Default?.ExitMethod(relu.Name);


            //LeakyReLU
            LeakyReLU leakyRelu = new LeakyReLU();
            RILogManager.Default?.EnterMethod(leakyRelu.Name);

            sw.Restart();
            gradArrayCpu = leakyRelu.Forward(inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            leakyRelu.Backward(gradArrayCpu);
            sw.Stop();

            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (leakyRelu.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = leakyRelu.Forward(inputArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                leakyRelu.Backward(gradArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }
            RILogManager.Default?.ExitMethod(leakyRelu.Name);


            //ReLuTanh
            ReLuTanh rth = new ReLuTanh();
            RILogManager.Default?.EnterMethod(rth.Name);

            sw.Restart();
            gradArrayCpu = rth.Forward(inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            rth.Backward(gradArrayCpu);
            sw.Stop();

            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (rth.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = rth.Forward(inputArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                rth.Backward(gradArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }
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
            NdArray[] gradImageArrayCpu = maxPooling.Forward(inputImageArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradImageArrayCpu[0].Grad = gradImageArrayCpu[0].Data;

            sw.Restart();
            maxPooling.Backward(gradImageArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (maxPooling.SetGpuEnable(true))
            {
                sw.Restart();
                maxPooling.Forward(inputImageArrayGpu);
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
            gradImageArrayCpu = avgPooling.Forward(inputImageArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradImageArrayCpu[0].Grad = gradImageArrayCpu[0].Data;

            sw.Restart();
            avgPooling.Backward(gradImageArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            RILogManager.Default?.ExitMethod(avgPooling.Name);


            //Conv2D
            Convolution2D conv2d = new Convolution2D(3, 3, 3);
            RILogManager.Default?.EnterMethod(conv2d.Name);

            sw.Restart();
            gradImageArrayCpu = conv2d.Forward(inputImageArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradImageArrayCpu[0].Grad = gradImageArrayCpu[0].Data;

            sw.Restart();
            conv2d.Backward(gradImageArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (conv2d.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradImageArrayGpu = conv2d.Forward(inputImageArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradImageArrayGpu[0].Grad = gradImageArrayGpu[0].Data;

                sw.Restart();
                conv2d.Backward(gradImageArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }
            RILogManager.Default?.ExitMethod(conv2d.Name);



            //Deconv2D
            Deconvolution2D deconv2d = new Deconvolution2D(3, 3, 3);
            RILogManager.Default?.EnterMethod(deconv2d.Name);

            sw.Restart();
            gradImageArrayCpu = deconv2d.Forward(inputImageArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradImageArrayCpu[0].Grad = gradImageArrayCpu[0].Data;

            sw.Restart();
            deconv2d.Backward(gradImageArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (deconv2d.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradImageArrayGpu = deconv2d.Forward(inputImageArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradImageArrayGpu[0].Grad = gradImageArrayGpu[0].Data;

                sw.Restart();
                deconv2d.Backward(gradImageArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }
            RILogManager.Default?.ExitMethod(deconv2d.Name);


            //Dropout
            Dropout dropout = new Dropout();
            RILogManager.Default?.EnterMethod(dropout.Name);

            sw.Restart();
            gradArrayCpu = dropout.Forward(inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            dropout.Backward(gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (dropout.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = dropout.Forward(inputArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                dropout.Backward(gradArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }
            RILogManager.Default?.ExitMethod(dropout.Name);





            //ArcSinH
            ArcSinH a = new ArcSinH();
            RILogManager.Default?.EnterMethod(a.Name);

            sw.Restart();
            gradArrayCpu = a.Forward(inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            a.Backward(gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (a.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = a.Forward(inputArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                a.Backward(gradArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }
            RILogManager.Default?.ExitMethod(a.Name);

            //ELU
            ELU e = new ELU();
            RILogManager.Default?.EnterMethod(e.Name);

            sw.Restart();
            gradArrayCpu = e.Forward(inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            e.Backward(gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            RILogManager.Default?.ExitMethod(e.Name);

            //LeakyReluShifted
            LeakyReLUShifted lrs = new LeakyReLUShifted();
            RILogManager.Default?.EnterMethod(lrs.Name);

            sw.Restart();
            gradArrayCpu = lrs.Forward(inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            lrs.Backward(gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (lrs.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = lrs.Forward(inputArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                lrs.Backward(gradArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }
            RILogManager.Default?.ExitMethod(lrs.Name);


            //Logistic
            LogisticFunction lf = new LogisticFunction();
            RILogManager.Default?.EnterMethod(lf.Name);

            sw.Restart();
            gradArrayCpu = lf.Forward(inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            lf.Backward(gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (lf.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = lf.Forward(inputArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                lf.Backward(gradArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }
            RILogManager.Default?.ExitMethod(lf.Name);


            //MaxMinusOne
            MaxMinusOne mmo = new MaxMinusOne();
            RILogManager.Default?.EnterMethod(mmo.Name);

            sw.Restart();
            gradArrayCpu = mmo.Forward(inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            mmo.Backward(gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (mmo.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = mmo.Forward(inputArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                mmo.Backward(gradArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }
            RILogManager.Default?.ExitMethod(mmo.Name);


            //ScaledELU
            ScaledELU se = new ScaledELU();
            RILogManager.Default?.EnterMethod(se.Name);

            sw.Restart();
            gradArrayCpu = se.Forward(inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            se.Backward(gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (se.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = se.Forward(inputArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                se.Backward(gradArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }
            RILogManager.Default?.ExitMethod(se.Name);


            //Sine
            Sine s = new Sine();
            RILogManager.Default?.EnterMethod(s.Name);

            sw.Restart();
            gradArrayCpu = s.Forward(inputArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Forward [Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            gradArrayCpu[0].Grad = gradArrayCpu[0].Data;

            sw.Restart();
            s.Backward(gradArrayCpu);
            sw.Stop();
            RILogManager.Default?.SendDebug("Backward[Cpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

            if (s.SetGpuEnable(true))
            {
                sw.Restart();
                NdArray[] gradArrayGpu = s.Forward(inputArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Forward [Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                gradArrayGpu[0].Grad = gradArrayGpu[0].Data;

                sw.Restart();
                s.Backward(gradArrayGpu);
                sw.Stop();
                RILogManager.Default?.SendDebug("Backward[Gpu] : " + (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");
            }
            RILogManager.Default?.ExitMethod(s.Name);



        }
    }
}
