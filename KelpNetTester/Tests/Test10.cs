﻿using System;
using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions.Container;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Noise;
using KelpNet.Loss;
using KelpNet.Optimizers;
using TestDataManager;
using VocabularyMaker;

namespace KelpNetTester.Tests
{
    //ChainerのRNNサンプルを再現
    //https://github.com/pfnet/chainer/tree/master/examples/ptb
    class Test10
    {
        const int N_EPOCH = 39;
        const int N_UNITS = 650;
        const int BATCH_SIZE = 20;
        const int BPROP_LEN = 35;
        const int GRAD_CLIP = 5;

        const string DOWNLOAD_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/";

        const string TRAIN_FILE = "ptb.train.txt";
        const string VALID_FILE = "ptb.valid.txt";
        const string TEST_FILE = "ptb.test.txt";

        public static void Run()
        {
            Console.WriteLine("Build Vocabulary.");

            Vocabulary vocabulary = new Vocabulary();

            string trainPath = InternetFileDownloader.Download(DOWNLOAD_URL + TRAIN_FILE, TRAIN_FILE);
            string validPath = InternetFileDownloader.Download(DOWNLOAD_URL + VALID_FILE, VALID_FILE);
            string testPath = InternetFileDownloader.Download(DOWNLOAD_URL + TEST_FILE, TEST_FILE);

            int[] trainData = vocabulary.LoadData(trainPath);
            int[] validData = vocabulary.LoadData(validPath);
            int[] testData = vocabulary.LoadData(testPath);

            int nVocab = vocabulary.Length;

            Console.WriteLine("Network Initilizing.");
            FunctionStack model = new FunctionStack(
                new EmbedID(nVocab, N_UNITS, name: "l1 EmbedID"),
                new Dropout(),
                new LSTM(N_UNITS, N_UNITS, name: "l2 LSTM"),
                new Dropout(),
                new LSTM(N_UNITS, N_UNITS, name: "l3 LSTM"),
                new Dropout(),
                new Linear(N_UNITS, nVocab, name: "l4 Linear")
            );

            //与えられたthresholdで頭打ちではなく、全パラメータのL2Normからレートを取り補正を行う
            GradientClipping gradientClipping = new GradientClipping(threshold: GRAD_CLIP);
            SGD sgd = new SGD(learningRate: 1);
            model.SetOptimizer(gradientClipping, sgd);

            Real wholeLen = trainData.Length;
            int jump = (int)Math.Floor(wholeLen / BATCH_SIZE);
            int epoch = 0;

            Stack<NdArray[]> backNdArrays = new Stack<NdArray[]>();

            Console.WriteLine("Train Start.");

            for (int i = 0; i < jump * N_EPOCH; i++)
            {
                NdArray x = new NdArray(new[] { 1 }, BATCH_SIZE);
                NdArray t = new NdArray(new[] { 1 }, BATCH_SIZE);

                for (int j = 0; j < BATCH_SIZE; j++)
                {
                    x.Data[j] = trainData[(int)((jump * j + i) % wholeLen)];
                    t.Data[j] = trainData[(int)((jump * j + i + 1) % wholeLen)];
                }

                NdArray[] result = model.Forward(x);
                Real sumLoss = new SoftmaxCrossEntropy().Evaluate(result, t);
                backNdArrays.Push(result);
                Console.WriteLine("[{0}/{1}] Loss: {2}", i + 1, jump, sumLoss);

                //Run truncated BPTT
                if ((i + 1) % BPROP_LEN == 0)
                {
                    for (int j = 0; backNdArrays.Count > 0; j++)
                    {
                        Console.WriteLine("backward" + backNdArrays.Count);
                        model.Backward(backNdArrays.Pop());
                    }

                    model.Update();
                    model.ResetState();
                }

                if ((i + 1) % jump == 0)
                {
                    epoch++;
                    Console.WriteLine("evaluate");
                    Console.WriteLine("validation perplexity: {0}", Evaluate(model, validData));

                    if (epoch >= 6)
                    {
                        sgd.LearningRate /= 1.2;
                        Console.WriteLine("learning rate =" + sgd.LearningRate);
                    }
                }
            }

            Console.WriteLine("test start");
            Console.WriteLine("test perplexity:" + Evaluate(model, testData));
        }

        static double Evaluate(FunctionStack model, int[] dataset)
        {
            FunctionStack predictModel = (FunctionStack)model.Clone();
            predictModel.ResetState();

            Real totalLoss = 0;
            long totalLossCount = 0;

            for (int i = 0; i < dataset.Length - 1; i++)
            {
                NdArray x = new NdArray(new[] { 1 }, BATCH_SIZE);
                NdArray t = new NdArray(new[] { 1 }, BATCH_SIZE);

                for (int j = 0; j < BATCH_SIZE; j++)
                {
                    x.Data[j] = dataset[j + i];
                    t.Data[j] = dataset[j + i + 1];
                }

                Real sumLoss = new SoftmaxCrossEntropy().Evaluate(predictModel.Forward(x), t);
                totalLoss += sumLoss;
                totalLossCount++;
            }

            //calc perplexity
            return Math.Exp(totalLoss / (totalLossCount - 1));
        }
    }
}
