using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using CaffemodelLoader;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Connections;
using KelpNet.Functions.Poolings;
using TestDataManager;

namespace KelpNetTester.Tests
{
    using ReflectSoftware.Insight;

    // Load ResNet and execute
    class Test17
    {
        private const string DOWNLOAD_URL_MEAN = "https://onedrive.live.com/download?cid=4006CBB8476FF777&resid=4006CBB8476FF777%2117894&authkey=%21AAFW2%2DFVoxeVRck";
        private const string DOWNLOAD_URL_50 = "https://onedrive.live.com/download?cid=4006CBB8476FF777&resid=4006CBB8476FF777%2117895&authkey=%21AAFW2%2DFVoxeVRck";
        private const string DOWNLOAD_URL_101 = "https://onedrive.live.com/download?cid=4006CBB8476FF777&resid=4006CBB8476FF777%2117896&authkey=%21AAFW2%2DFVoxeVRck";
        private const string DOWNLOAD_URL_152 = "https://onedrive.live.com/download?cid=4006CBB8476FF777&resid=4006CBB8476FF777%2117897&authkey=%21AAFW2%2DFVoxeVRck";

        private const string MODEL_FILE_MEAN = "ResNet_mean.binaryproto";
        private const string MODEL_FILE_50 = "ResNet-50-model.caffemodel";
        private const string MODEL_FILE_101 = "ResNet-101-model.caffemodel";
        private const string MODEL_FILE_152 = "ResNet-152-model.caffemodel";

        private static readonly string[] Urls = { DOWNLOAD_URL_50, DOWNLOAD_URL_101, DOWNLOAD_URL_152 };
        private static readonly string[] FileNames = { MODEL_FILE_50, MODEL_FILE_101, MODEL_FILE_152 };

        private const string CLASS_LIST_PATH = "Data/synset_words.txt";

        public enum ResnetModel
        {
            ResNet50,
            ResNet101,
            ResNet152,
        }

        public static void Run(ResnetModel modelType)
        {
            OpenFileDialog ofd = new OpenFileDialog { Filter = "Image files(*.jpg;*.png;*.gif;*.bmp)|*.jpg;*.png;*.gif;*.bmp|All files(*.*)|*.*" };

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                int resnetId = (int)modelType;

                RILogManager.Default?.SendDebug("Mean Loading.");
                string meanFilePath = InternetFileDownloader.Download(DOWNLOAD_URL_MEAN, MODEL_FILE_MEAN);
                NdArray mean = CaffemodelDataLoader.ReadBinary(meanFilePath);

                RILogManager.Default?.SendDebug("Model Loading.");
                string modelFilePath = InternetFileDownloader.Download(Urls[resnetId], FileNames[resnetId]);
                FunctionDictionary nn = CaffemodelDataLoader.LoadNetWork(true, modelFilePath);
                string[] classList = File.ReadAllLines(CLASS_LIST_PATH);

                // Initialize the GPU
                foreach (FunctionStack resNetFunctionBlock in nn.FunctionBlocks)
                {
                    SwitchGPU(resNetFunctionBlock);
                }

                RILogManager.Default?.SendDebug("Model Loading done.");

                do
                {
                    // Set the resolution to 224px x 224px x 3ch before entering the network
                    Bitmap baseImage = new Bitmap(ofd.FileName);
                    Bitmap resultImage = new Bitmap(224, 224, PixelFormat.Format24bppRgb);
                    Graphics g = Graphics.FromImage(resultImage);
                    g.InterpolationMode = InterpolationMode.Bilinear;
                    g.DrawImage(baseImage, 0, 0, 224, 224);
                    g.Dispose();

                    NdArray imageArray = NdArrayConverter.Image2NdArray(resultImage, false, true);
                    imageArray -= mean;
                    imageArray.ParentFunc = null;

                    RILogManager.Default?.SendDebug("Start predict.");
                    Stopwatch sw = Stopwatch.StartNew();
                    NdArray result = nn.Predict(true, imageArray)[0];
                    sw.Stop();

                    RILogManager.Default?.SendDebug("Result Time : " +
                                      (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") +
                                      "μｓ");

                    int maxIndex = Array.IndexOf(result.Data, result.Data.Max());
                    RILogManager.Default?.SendDebug("[" + result.Data[maxIndex] + "] : " + classList[maxIndex]);
                } 
                while (ofd.ShowDialog() == DialogResult.OK);
            }
        }

        static void SwitchGPU(FunctionStack functionStack)
        {
            foreach (Function function in functionStack.Functions)
            {
                if (function is Convolution2D || function is Linear || function is MaxPooling)
                {
                    ((IParallelizable)function).SetGpuEnable(true);
                }

                if (function is SplitFunction)
                {
                    SplitFunction splitFunction = (SplitFunction)function;
                    foreach (var t in splitFunction.SplitedFunctions)
                    {
                        SwitchGPU(t);
                    }
                }
            }

            // Compact layer on block basis
            functionStack.Compress();
        }
    }
}
