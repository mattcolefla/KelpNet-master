using System;
using System.Collections.Generic;
using System.Text;
using TestDataManager;

namespace CIFARLoader
{
    /// <summary>   A cifar data loader. </summary>
    public class CIFARDataLoader
    {
        /// <summary>   URL of the download. </summary>
        const string DOWNLOAD_URL = "http://www.cs.toronto.edu/~kriz/";

        /// <summary>   The cifar 10. </summary>
        const string CIFAR10 = "cifar-10-binary.tar.gz";
        /// <summary>   List of names of the cifar 10 trains. </summary>
        private readonly string[] CIFAR10TrainNames =
        {
            "cifar-10-batches-bin/data_batch_1.bin",
            "cifar-10-batches-bin/data_batch_2.bin",
            "cifar-10-batches-bin/data_batch_3.bin",
            "cifar-10-batches-bin/data_batch_4.bin",
            "cifar-10-batches-bin/data_batch_5.bin",
        };
        /// <summary>   Name of the cifar 10 test. </summary>
        private readonly string CIFAR10TestName = "cifar-10-batches-bin/test_batch.bin";
        /// <summary>   Number of cifar 10 data. </summary>
        private const int CIFAR10_DATA_COUNT = 10000;

        /// <summary>   The cifar 100. </summary>
        const string CIFAR100 = "cifar-100-binary.tar.gz";
        /// <summary>   Name of the cifar 100 train. </summary>
        private readonly string CIFAR100TrainName = "cifar-100-binary/train.bin";
        /// <summary>   Name of the cifar 100 test. </summary>
        private readonly string CIFAR100TestName = "cifar-100-binary/test.bin";
        /// <summary>   Number of cifar 100 data. </summary>
        private const int CIFAR100_DATA_COUNT = 50000;
        /// <summary>   Number of cifar 100 test data. </summary>
        private const int CIFAR100_TEST_DATA_COUNT = 10000;


        /// <summary>   List of names of the labels. </summary>
        public string[] LabelNames;
        /// <summary>   List of names of the fine labels. </summary>
        public string[] FineLabelNames;

        /// <summary>   The train label. </summary>
        public byte[] TrainLabel;
        /// <summary>   The train fine label. </summary>
        public byte[] TrainFineLabel;
        /// <summary>   Information describing the train. </summary>
        public byte[][] TrainData;

        /// <summary>   The test label. </summary>
        public byte[] TestLabel;
        /// <summary>   The test fine label. </summary>
        public byte[] TestFineLabel;
        /// <summary>   Information describing the test. </summary>
        public byte[][] TestData;

        /// <summary>   Size of the label. </summary>
        private const int LABEL_SIZE = 1;
        /// <summary>   Size of the data. </summary>
        private const int DATA_SIZE = 3072;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Initializes a new instance of the CIFARLoader.CIFARDataLoader class. </summary>
        ///
        /// <param name="isCifar100">   (Optional) True if this object is cifar 100. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public CIFARDataLoader(bool isCifar100 = false)
        {
            if (!isCifar100)
            {
                string cifar10Path = InternetFileDownloader.Download(DOWNLOAD_URL + CIFAR10, CIFAR10);
                Dictionary<string, byte[]> data = Tar.GetExtractedStreams(cifar10Path);

                LabelNames = Encoding.ASCII.GetString(data["cifar-10-batches-bin/batches.meta.txt"]).Split(new[] {'\n'}, StringSplitOptions.RemoveEmptyEntries);

                List<byte> trainLabel = new List<byte>();
                List<byte[]> trainData = new List<byte[]>();

                foreach (var t in CIFAR10TrainNames)
                {
                    for (int j = 0; j < CIFAR10_DATA_COUNT; j++)
                    {
                        trainLabel.Add(data[t][j * (DATA_SIZE + LABEL_SIZE)]);
                        byte[] tmpArray = new byte[DATA_SIZE];
                        Array.Copy(data[t], j * (DATA_SIZE + LABEL_SIZE) + LABEL_SIZE, tmpArray, 0, tmpArray.Length);
                        trainData.Add(tmpArray);
                    }
                }

                TrainLabel = trainLabel.ToArray();
                TrainData = trainData.ToArray();

                List<byte> testLabel = new List<byte>();
                List<byte[]> testData = new List<byte[]>();

                for (int j = 0; j < CIFAR10_DATA_COUNT; j++)
                {
                    testLabel.Add(data[CIFAR10TestName][j * (DATA_SIZE + LABEL_SIZE)]);
                    byte[] tmpArray = new byte[DATA_SIZE];
                    Array.Copy(data[CIFAR10TestName], j * (DATA_SIZE + LABEL_SIZE) + LABEL_SIZE, tmpArray, 0, tmpArray.Length);
                    testData.Add(tmpArray);
                }

                TestLabel = testLabel.ToArray();
                TestData = testData.ToArray();
            }
            else
            {
                string cifar100Path = InternetFileDownloader.Download(DOWNLOAD_URL + CIFAR100, CIFAR100);
                Dictionary<string, byte[]> data = Tar.GetExtractedStreams(cifar100Path);

                LabelNames = Encoding.ASCII.GetString(data["cifar-100-binary/coarse_label_names.txt"]).Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);
                FineLabelNames = Encoding.ASCII.GetString(data["cifar-100-binary/fine_label_names.txt"]).Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);

                List<byte> trainLabel = new List<byte>();
                List<byte> trainFineLabel = new List<byte>();
                List<byte[]> trainData = new List<byte[]>();

                for (int j = 0; j < CIFAR100_DATA_COUNT; j++)
                {
                    trainLabel.Add(data[CIFAR100TrainName][j * (DATA_SIZE + LABEL_SIZE + LABEL_SIZE)]);
                    trainFineLabel.Add(data[CIFAR100TrainName][j * (DATA_SIZE + LABEL_SIZE + LABEL_SIZE) + LABEL_SIZE]);
                    byte[] tmpArray = new byte[DATA_SIZE];
                    Array.Copy(data[CIFAR100TrainName], j * (DATA_SIZE + LABEL_SIZE + LABEL_SIZE) + LABEL_SIZE + LABEL_SIZE, tmpArray, 0,
                        tmpArray.Length);
                    trainData.Add(tmpArray);
                }

                TrainLabel = trainLabel.ToArray();
                TrainFineLabel = trainFineLabel.ToArray();
                TrainData = trainData.ToArray();

                List<byte> testLabel = new List<byte>();
                List<byte> testFineLabel = new List<byte>();
                List<byte[]> testData = new List<byte[]>();

                for (int j = 0; j < CIFAR100_TEST_DATA_COUNT; j++)
                {
                    testLabel.Add(data[CIFAR100TestName][j * (DATA_SIZE + LABEL_SIZE + LABEL_SIZE)]);
                    testFineLabel.Add(data[CIFAR100TestName][j * (DATA_SIZE + LABEL_SIZE + LABEL_SIZE) + LABEL_SIZE]);
                    byte[] tmpArray = new byte[DATA_SIZE];
                    Array.Copy(data[CIFAR100TestName], j * (DATA_SIZE + LABEL_SIZE + LABEL_SIZE) + LABEL_SIZE + LABEL_SIZE, tmpArray, 0, tmpArray.Length);
                    testData.Add(tmpArray);
                }

                TestLabel = testLabel.ToArray();
                TestFineLabel = testFineLabel.ToArray();
                TestData = testData.ToArray();
            }
        }
    }
}
