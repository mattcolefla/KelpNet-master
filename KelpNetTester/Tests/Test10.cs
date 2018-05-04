using System;
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
    using System.IO;
    using System.Threading;
    using HdrHistogram;
    using ReflectSoftware.Insight;

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

        private const string LogPath = "Histogram.log";
        private static HistogramLogWriter _logWriter;
        private static FileStream _outputStream;
        private static int _isCompleted = -1;
        private static bool _running;


        public static void Run()
        {
            _outputStream = File.Create(LogPath);
            
            _logWriter = new HistogramLogWriter(_outputStream);
            _logWriter.Write(DateTime.Now);

            var recorder = HistogramFactory
                .With64BitBucketSize()                  
                ?.WithValuesFrom(1)                      
                ?.WithValuesUpTo(2345678912345)  
                ?.WithPrecisionOf(3)                     
                ?.WithThreadSafeWrites()                 
                ?.WithThreadSafeReads()                  
                ?.Create();
            
            var accumulatingHistogram = new LongHistogram(2345678912345, 3);

            var size = accumulatingHistogram.GetEstimatedFootprintInBytes();
            RILogManager.Default?.SendDebug("Histogram size = {0} bytes ({1:F2} MB)", size, size / 1024.0 / 1024.0);


            RILogManager.Default?.SendDebug("Recorded latencies [in system clock ticks]");
            accumulatingHistogram.OutputPercentileDistribution(Console.Out, outputValueUnitScalingRatio: OutputScalingFactor.None, useCsvFormat: true);
            Console.WriteLine();

            RILogManager.Default?.SendDebug("Recorded latencies [in usec]");
            accumulatingHistogram.OutputPercentileDistribution(Console.Out, outputValueUnitScalingRatio: OutputScalingFactor.TimeStampToMicroseconds, useCsvFormat: true);
            Console.WriteLine();

            RILogManager.Default?.SendDebug("Recorded latencies [in msec]");
            accumulatingHistogram.OutputPercentileDistribution(Console.Out, outputValueUnitScalingRatio: OutputScalingFactor.TimeStampToMilliseconds, useCsvFormat: true);
            Console.WriteLine();

            RILogManager.Default?.SendDebug("Recorded latencies [in sec]");
            accumulatingHistogram.OutputPercentileDistribution(Console.Out, outputValueUnitScalingRatio: OutputScalingFactor.TimeStampToSeconds, useCsvFormat: true);

            DocumentResults(accumulatingHistogram, recorder);

            RILogManager.Default?.SendDebug("Build Vocabulary.");

            DocumentResults(accumulatingHistogram, recorder);

            Vocabulary vocabulary = new Vocabulary();

            DocumentResults(accumulatingHistogram, recorder);

            string trainPath = InternetFileDownloader.Download(DOWNLOAD_URL + TRAIN_FILE, TRAIN_FILE);
            DocumentResults(accumulatingHistogram, recorder);

            string validPath = InternetFileDownloader.Download(DOWNLOAD_URL + VALID_FILE, VALID_FILE);
            DocumentResults(accumulatingHistogram, recorder);

            string testPath = InternetFileDownloader.Download(DOWNLOAD_URL + TEST_FILE, TEST_FILE);
            DocumentResults(accumulatingHistogram, recorder);


            int[] trainData = vocabulary.LoadData(trainPath);
            DocumentResults(accumulatingHistogram, recorder);

            int[] validData = vocabulary.LoadData(validPath);
            DocumentResults(accumulatingHistogram, recorder);

            int[] testData = vocabulary.LoadData(testPath);
            DocumentResults(accumulatingHistogram, recorder);

            int nVocab = vocabulary.Length;

            RILogManager.Default?.SendDebug("Network Initializing.");
            FunctionStack model = new FunctionStack(
                new EmbedID(nVocab, N_UNITS, name: "l1 EmbedID"),
                new Dropout(),
                new LSTM(N_UNITS, N_UNITS, name: "l2 LSTM"),
                new Dropout(),
                new LSTM(N_UNITS, N_UNITS, name: "l3 LSTM"),
                new Dropout(),
                new Linear(N_UNITS, nVocab, name: "l4 Linear")
            );
            DocumentResults(accumulatingHistogram, recorder);

            // Do not cease at the given threshold, correct the rate by taking the rate from L2Norm of all parameters
            GradientClipping gradientClipping = new GradientClipping(threshold: GRAD_CLIP);
            SGD sgd = new SGD(learningRate: 1);
            model.SetOptimizer(gradientClipping, sgd);
            DocumentResults(accumulatingHistogram, recorder);

            Real wholeLen = trainData.Length;
            int jump = (int)Math.Floor(wholeLen / BATCH_SIZE);
            int epoch = 0;

            Stack<NdArray[]> backNdArrays = new Stack<NdArray[]>();

            RILogManager.Default?.SendDebug("Train Start.");
            double dVal;
            NdArray x = new NdArray(new[] { 1 }, BATCH_SIZE);
            NdArray t = new NdArray(new[] { 1 }, BATCH_SIZE);

            for (int i = 0; i < jump * N_EPOCH; i++)
            {
                for (int j = 0; j < BATCH_SIZE; j++)
                {
                    x.Data[j] = trainData[(int)((jump * j + i) % wholeLen)];
                    t.Data[j] = trainData[(int)((jump * j + i + 1) % wholeLen)];
                }

                NdArray[] result = model.Forward(x);
                Real sumLoss = new SoftmaxCrossEntropy().Evaluate(result, t);
                backNdArrays.Push(result);
                RILogManager.Default?.SendDebug("[{0}/{1}] Loss: {2}", i + 1, jump, sumLoss);

                //Run truncated BPTT
                if ((i + 1) % BPROP_LEN == 0)
                {
                    for (int j = 0; backNdArrays.Count > 0; j++)
                    {
                        RILogManager.Default?.SendDebug("backward" + backNdArrays.Count);
                        model.Backward(backNdArrays.Pop());
                    }

                    model.Update();
                    model.ResetState();
                }

                if ((i + 1) % jump == 0)
                {
                    epoch++;
                    RILogManager.Default?.SendDebug("evaluate");
                    dVal = Evaluate(model, validData);
                    RILogManager.Default?.SendDebug($"validation perplexity: {dVal}");

                    if (epoch >= 6)
                    {
                        sgd.LearningRate /= 1.2;
                        RILogManager.Default?.SendDebug("learning rate =" + sgd.LearningRate);
                    }
                }
                DocumentResults(accumulatingHistogram, recorder);
            }

            RILogManager.Default?.SendDebug("test start");
            dVal = Evaluate(model, testData);
            RILogManager.Default?.SendDebug("test perplexity:" + dVal);
            DocumentResults(accumulatingHistogram, recorder);

            _logWriter.Dispose();
            _outputStream.Dispose();


            RILogManager.Default?.SendDebug("Log contents");
            RILogManager.Default?.SendDebug(File.ReadAllText(LogPath));
            Console.WriteLine();
            RILogManager.Default?.SendDebug("Percentile distribution (values reported in milliseconds)");
            accumulatingHistogram.OutputPercentileDistribution(Console.Out, outputValueUnitScalingRatio: OutputScalingFactor.TimeStampToMilliseconds, useCsvFormat: true);

            RILogManager.Default?.SendDebug("Mean: " + BytesToString(accumulatingHistogram.GetMean()) + ", StdDev: " +
                                            BytesToString(accumulatingHistogram.GetStdDeviation()));
        }

        static void DocumentResults(LongHistogram accumulatingHistogram, Recorder recorder)
        {
            recorder?.RecordValue(GC.GetTotalMemory(false));
            var histogram = recorder?.GetIntervalHistogram();
            accumulatingHistogram?.Add(histogram);
            _logWriter?.Append(histogram);
            RILogManager.Default?.SendDebug($"Accumulated.TotalCount = {accumulatingHistogram.TotalCount,10:G}.");
            RILogManager.Default?.SendDebug("Mean: " + BytesToString(accumulatingHistogram.GetMean()) + ", StdDev: " +
                                            BytesToString(accumulatingHistogram.GetStdDeviation()));

        }

        static double Evaluate(FunctionStack model, int[] dataset)
        {
            FunctionStack predictModel = (FunctionStack)model.Clone();
            predictModel.ResetState();

            Real totalLoss = 0;
            long totalLossCount = 0;
            NdArray x = new NdArray(new[] { 1 }, BATCH_SIZE);
            NdArray t = new NdArray(new[] { 1 }, BATCH_SIZE);

            for (int i = 0; i < dataset.Length - 1; i++)
            {
                for (int j = 0; j < BATCH_SIZE; j++)
                {
                    x.Data[j] = dataset[j + i];
                    t.Data[j] = dataset[j + i + 1];
                }

                RILogManager.Default?.SendDebug("validating Cross Entropy " + i);

                Real sumLoss = new SoftmaxCrossEntropy().Evaluate(predictModel.Forward(x), t);
                totalLoss += sumLoss;
                totalLossCount++;
            }

            //calc perplexity
            return Math.Exp(totalLoss / (totalLossCount - 1));
        }

        static String BytesToString(double byteCount)
        {
            string[] suf = { "B", "KB", "MB", "GB", "TB", "PB", "EB" }; //Longs run out around EB
            if (byteCount == 0.0)
                return "0" + suf[0];
            long bytes = Convert.ToInt64(Math.Abs(byteCount));
            int place = Convert.ToInt32(Math.Floor(Math.Log(bytes, 1024)));
            double num = Math.Round(bytes / Math.Pow(1024, place), 1);
            return (Math.Sign(byteCount) * num).ToString() + suf[place];
        }

    }
}
