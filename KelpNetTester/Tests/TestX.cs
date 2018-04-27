﻿using System;
using KelpNet.Common;
using KelpNet.Functions.Connections;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    using ReflectSoftware.Insight;

    //Linear split test
    class TestX
    {
        public static void Run()
        {
            //Align Weight before and after splitting
            Real[,] testWeightValues = {
                            {-0.02690255, 0.08830735, -0.02041466, -0.0431439, -0.07749002},
                            {-0.06963444, -0.03971611, 0.0597842, 0.08824182, -0.06649109},
                            {-0.04966073, -0.04697048, -0.02235234, -0.09396666, 0.073189},
                            {0.06563969, 0.04446745, -0.07192299, 0.06784364, 0.09575776},
                            {0.05012317, -0.08874852, -0.05977172, -0.05910181, -0.06009106},
                            {-0.05200623, -0.09679124, 0.02159978, -0.08058041, -0.01340541},
                            {-0.0254951, 0.09963084, 0.00936683, -0.08179696, 0.09604459},
                            {-0.0732494, 0.07253634, 0.05981455, -0.01007657, -0.02992892},
                            {-0.06818873, -0.02579817, 0.06767359, -0.03379837, -0.04880046},
                            {-0.06429326, -0.08964688, -0.0960066, -0.00286683, -0.05761427},
                            {-0.0454098, 0.07809167, -0.05030088, -0.02533244, -0.02322736},
                            {-0.00866754, -0.03614252, 0.05237325, 0.06478979, -0.03599609},
                            {-0.01789357, -0.04479434, -0.05765592, 0.03237658, -0.06403019},
                            {-0.02421552, 0.05533903, -0.08627617, 0.094624, 0.03319318},
                            {0.02328842, -0.08234859, -0.07979888, 0.01439688, -0.03267198},
                            {-0.07128382, 0.08531934, 0.07180037, 0.04772871, -0.08938966},
                            {0.09431138, 0.02094762, 0.04443646, 0.07653841, 0.02028433},
                            {0.01844446, -0.08441339, 0.01957355, 0.04430714, -0.03080243},
                            {-0.0261334, -0.03794889, -0.00638074, 0.07278767, -0.02165155},
                            {0.08390063, -0.03253863, 0.0311571, 0.08088892, -0.07267931}
            };

            Real[][,] testJaggWeightValues = {
                new Real[,] {{-0.02690255, 0.08830735, -0.02041466, -0.0431439, -0.07749002},
                             {-0.06963444, -0.03971611, 0.0597842, 0.08824182, -0.06649109},
                             {-0.04966073, -0.04697048, -0.02235234, -0.09396666, 0.073189},
                             {0.06563969, 0.04446745, -0.07192299, 0.06784364, 0.09575776},
                             {0.05012317, -0.08874852, -0.05977172, -0.05910181, -0.06009106}},
                new Real[,] {{-0.05200623, -0.09679124, 0.02159978, -0.08058041, -0.01340541},
                             {-0.0254951, 0.09963084, 0.00936683, -0.08179696, 0.09604459},
                             {-0.0732494, 0.07253634, 0.05981455, -0.01007657, -0.02992892},
                             {-0.06818873, -0.02579817, 0.06767359, -0.03379837, -0.04880046},
                             {-0.06429326, -0.08964688, -0.0960066, -0.00286683, -0.05761427}},
                new Real[,] {{-0.0454098, 0.07809167, -0.05030088, -0.02533244, -0.02322736},
                             {-0.00866754, -0.03614252, 0.05237325, 0.06478979, -0.03599609},
                             {-0.01789357, -0.04479434, -0.05765592, 0.03237658, -0.06403019},
                             {-0.02421552, 0.05533903, -0.08627617, 0.094624, 0.03319318},
                             {0.02328842, -0.08234859, -0.07979888, 0.01439688, -0.03267198}},
                new Real[,] {{-0.07128382, 0.08531934, 0.07180037, 0.04772871, -0.08938966},
                             {0.09431138, 0.02094762, 0.04443646, 0.07653841, 0.02028433},
                             {0.01844446, -0.08441339, 0.01957355, 0.04430714, -0.03080243},
                             {-0.0261334, -0.03794889, -0.00638074, 0.07278767, -0.02165155},
                             {0.08390063, -0.03253863, 0.0311571, 0.08088892, -0.07267931}}
            };

            Linear l0 = new Linear(5, 20, initialW: testWeightValues, name: "l0");
            Linear l1 = new Linear(5, 5, initialW: testJaggWeightValues[0], name: "l1");
            Linear l2 = new Linear(5, 5, initialW: testJaggWeightValues[1], name: "l2");
            Linear l3 = new Linear(5, 5, initialW: testJaggWeightValues[2], name: "l3");
            Linear l4 = new Linear(5, 5, initialW: testJaggWeightValues[3], name: "l4");

            l0.SetOptimizer(new SGD());

            SGD sgd = new SGD();
            l1.SetOptimizer(sgd);
            l2.SetOptimizer(sgd);
            l3.SetOptimizer(sgd);
            l4.SetOptimizer(sgd);


            //Input is equivalent, but Grad is added and it is divided
            Real[] testValue = { 0.01618112, -0.08296648, -0.05545357, 0.00389254, -0.05727582 };
            NdArray testInputValuesA = new NdArray(testValue);
            NdArray testInputValuesB = new NdArray(testValue);

            RILogManager.Default?.SendDebug("l0 for");
            NdArray[] l0Result = l0.Forward(testInputValuesA);
            RILogManager.Default?.SendDebug(l0Result.ToString());

            RILogManager.Default?.SendDebug("l1 for");
            NdArray[] l1Result = l1.Forward(testInputValuesB);
            RILogManager.Default?.SendDebug(l1Result.ToString());

            RILogManager.Default?.SendDebug("l2 for");
            NdArray[] l2Result = l2.Forward(testInputValuesB);
            RILogManager.Default?.SendDebug(l2Result.ToString());

            RILogManager.Default?.SendDebug("l3 for");
            NdArray[] l3Result = l3.Forward(testInputValuesB);
            RILogManager.Default?.SendDebug(l3Result.ToString());

            RILogManager.Default?.SendDebug("l4 for");
            NdArray[] l4Result = l4.Forward(testInputValuesB);
            RILogManager.Default?.SendDebug(l4Result.ToString());


            //Create an appropriate Grad value
            l0Result[0].Grad = new Real[]
                                    {
                                        -2.42022760e-02, 5.02482988e-04, 2.52015481e-04, 8.08797951e-04, -7.19293347e-03,
                                        1.40045900e-04, 7.09874439e-05, 2.07651625e-04, 3.80124636e-02, -8.87162634e-04,
                                        -4.64874669e-04, -1.40792923e-03, -4.12280299e-02, -3.36557830e-04, -1.50323089e-04,
                                        -4.70047118e-04, 3.61101292e-02, -7.12957408e-04, -3.63163825e-04, -1.12809543e-03
                                    };

            l1Result[0].Grad = new Real[] {-2.42022760e-02,5.02482988e-04, 2.52015481e-04,8.08797951e-04,- 7.19293347e-03};
            l2Result[0].Grad = new Real[] {1.40045900e-04, 7.09874439e-05, 2.07651625e-04, 3.80124636e-02, -8.87162634e-04};
            l3Result[0].Grad = new Real[] {-4.64874669e-04, -1.40792923e-03, -4.12280299e-02, -3.36557830e-04, -1.50323089e-04};
            l4Result[0].Grad = new Real[] {-4.70047118e-04, 3.61101292e-02, -7.12957408e-04, -3.63163825e-04, -1.12809543e-03};
            

            //Backward
            l0.Backward(l0Result);
            l1.Backward(l1Result);
            l2.Backward(l2Result);
            l3.Backward(l3Result);
            l4.Backward(l4Result);

            RILogManager.Default?.SendDebug("l0 back");
            RILogManager.Default?.SendDebug(testInputValuesA.ToString("Grad"));
            RILogManager.Default?.SendDebug("l1-l4 sum back");
            RILogManager.Default?.SendDebug(testInputValuesB.ToString("Grad"));

            l0.Update();  //Although the format is irregular, since 10 contains SGD
            sgd.Update(); // Use stochastic gradient descent as the optimizer

            RILogManager.Default?.SendDebug("l0 Weight");
            RILogManager.Default?.SendDebug(l0.Weight.ToString());
            RILogManager.Default?.SendDebug("l1 Weight");
            RILogManager.Default?.SendDebug(l1.Weight.ToString());
            RILogManager.Default?.SendDebug("l0 Bias");
            RILogManager.Default?.SendDebug(l0.Bias.ToString());
            RILogManager.Default?.SendDebug("l1 Bias");
            RILogManager.Default?.SendDebug(l1.Bias.ToString());
        }
    }
}
