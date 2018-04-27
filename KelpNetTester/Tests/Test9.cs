using System;
using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions.Container;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;
using TestDataManager;
using VocabularyMaker;

namespace KelpNetTester.Tests
{
    using System.Diagnostics.CodeAnalysis;
    using ReflectSoftware.Insight;

    //RNNLM with Simple RNN
    //From 'practical deep learning by Chainer'（ISBN 978-4-274-21934-4）
    [SuppressMessage("ReSharper", "LocalizableElement")]
    class Test9
    {
        const int TRAINING_EPOCHS = 5;
        const int N_UNITS = 100;
        const string DOWNLOAD_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/";
        const string TRAIN_FILE = "ptb.train.txt";
        const string TEST_FILE = "ptb.test.txt";

        public static void Run()
        {
            RILogManager.Default?.SendDebug("Building Vocabulary.");

            Vocabulary vocabulary = new Vocabulary();
            string trainPath = InternetFileDownloader.Download(DOWNLOAD_URL + TRAIN_FILE, TRAIN_FILE);
            string testPath = InternetFileDownloader.Download(DOWNLOAD_URL + TEST_FILE, TEST_FILE);
            int[] trainData = vocabulary.LoadData(trainPath);
            int[] testData = vocabulary.LoadData(testPath);
            int nVocab = vocabulary.Length;

            RILogManager.Default?.SendDebug("Network Initializing.");
            FunctionStack model = new FunctionStack(
                new EmbedID(nVocab, N_UNITS, name: "l1 EmbedID"),
                new Linear(N_UNITS, N_UNITS, name: "l2 Linear"),
                new Tanh("l2 Tanh"),
                new Linear(N_UNITS, nVocab, name: "l3 Linear"),
                new Softmax("l3 Softmax")
            );

            model.SetOptimizer(new Adam());

            List<int> s = new List<int>();

            RILogManager.Default?.SendDebug("Train Start.");
            SoftmaxCrossEntropy softmaxCrossEntropy = new SoftmaxCrossEntropy();
            for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++)
            {
                for (int pos = 0; pos < trainData.Length; pos++)
                {
                    NdArray h = new NdArray(new Real[N_UNITS]);
                    int id = trainData[pos];
                    s.Add(id);

                    if (id == vocabulary.EosID)
                    {
                        Real accumloss = 0;
                        Stack<NdArray> tmp = new Stack<NdArray>();

                        for (int i = 0; i < s.Count; i++)
                        {
                            int tx = i == s.Count - 1 ? vocabulary.EosID : s[i + 1];

                            //l1 EmbedID
                            NdArray l1 = model.Functions[0].Forward(s[i])[0];

                            //l2 Linear
                            NdArray l2 = model.Functions[1].Forward(h)[0];

                            //Add
                            NdArray xK = l1 + l2;

                            //l2 Tanh
                            h = model.Functions[2].Forward(xK)[0];

                            //l3 Linear
                            NdArray h2 = model.Functions[3].Forward(h)[0];

                            Real loss = softmaxCrossEntropy.Evaluate(h2, tx);
                            tmp.Push(h2);
                            accumloss += loss;
                        }

                        RILogManager.Default?.SendDebug(accumloss.ToString());

                        for (int i = 0; i < s.Count; i++)
                        {
                            model.Backward(tmp.Pop());
                        }

                        model.Update();
                        s.Clear();
                    }

                    if (pos % 100 == 0)
                    {
                        RILogManager.Default?.SendDebug(pos + "/" + trainData.Length + " finished");
                    }
                }
            }

            RILogManager.Default?.SendDebug("Test Start.");

            Real sum = 0;
            int wnum = 0;
            List<int> ts = new List<int>();
            bool unkWord = false;

            for (int pos = 0; pos < 1000; pos++)
            {
                int id = testData[pos];
                ts.Add(id);

                if (id > trainData.Length)
                {
                    unkWord = true;
                }

                if (id == vocabulary.EosID)
                {
                    if (!unkWord)
                    {
                        RILogManager.Default?.SendDebug("pos" + pos);
                        RILogManager.Default?.SendDebug("tsLen" + ts.Count);
                        RILogManager.Default?.SendDebug("sum" + sum);
                        RILogManager.Default?.SendDebug("wnum" + wnum);

                        sum += CalPs(model, ts);
                        wnum += ts.Count - 1;
                    }
                    else
                    {
                        unkWord = false;
                    }

                    ts.Clear();
                }
            }

            RILogManager.Default?.SendDebug(Math.Pow(2.0, sum / wnum).ToString());
        }

        static Real CalPs(FunctionStack model, List<int> s)
        {
            Real sum = 0;

            NdArray h = new NdArray(new Real[N_UNITS]);

            for (int i = 1; i < s.Count; i++)
            {
                //l1 Linear
                NdArray xK = model.Functions[0].Forward(s[i])[0];

                //l2 Linear
                NdArray l2 = model.Functions[1].Forward(h)[0];
                for (int j = 0; j < xK.Data.Length; j++)
                {
                    xK.Data[j] += l2.Data[j];
                }

                //l2 Tanh
                h = model.Functions[2].Forward(xK)[0];

                //l3 Softmax(l3 Linear)
                NdArray yv = model.Functions[4].Forward(model.Functions[3].Forward(h))[0];
                Real pi = yv.Data[s[i - 1]];
                sum -= Math.Log(pi, 2);
            }

            return sum;
        }
    }
}

