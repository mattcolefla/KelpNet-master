using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
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

    // Test that reads VGG 16 of Caffe model and makes image classification
    class Test15
    {
        private const string DOWNLOAD_URL = "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel";
        private const string MODEL_FILE = "VGG_ILSVRC_16_layers.caffemodel";
        private const string CLASS_LIST_PATH = "Data/synset_words.txt";

        public static void Run()
        {
            OpenFileDialog ofd = new OpenFileDialog { Filter = "Image Files(*.jpg;*.png;*.gif;*.bmp)|*.jpg;*.png;*.gif;*.bmp|All Files(*.*)|*.*" };

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                RILogManager.Default?.SendDebug("Model Loading.");
                string modelFilePath = InternetFileDownloader.Download(DOWNLOAD_URL, MODEL_FILE);
                List<Function> vgg16Net = CaffemodelDataLoader.ModelLoad(modelFilePath);
                string[] classList = File.ReadAllLines(CLASS_LIST_PATH);

                // Initialize the GPU
                for (int i = 0; i < vgg16Net.Count - 1; i++)
                {
                    if (vgg16Net[i] is Convolution2D || vgg16Net[i] is Linear || vgg16Net[i] is MaxPooling)
                    {
                        ((IParallelizable) vgg16Net[i]).SetGpuEnable(true);
                    }
                }

                FunctionStack nn = new FunctionStack(vgg16Net.ToArray());

                // compress layer
                nn.Compress();

                RILogManager.Default?.SendDebug("Model Loading done.");

                do
                {
                    // Set the resolution to 224px x 224px x 3ch before entering the network
                    Bitmap baseImage = new Bitmap(ofd.FileName);
                    Bitmap resultImage = new Bitmap(224, 224, PixelFormat.Format24bppRgb);
                    Graphics g = Graphics.FromImage(resultImage);
                    g.DrawImage(baseImage, 0, 0, 224, 224);
                    g.Dispose();

                    Real[] bias = { -123.68, -116.779, -103.939 }; // The channel order of the correction value follows the input image
                    NdArray imageArray = NdArrayConverter.Image2NdArray(resultImage, false, true, bias);

                    RILogManager.Default?.SendDebug("Start predict.");
                    Stopwatch sw = Stopwatch.StartNew();
                    NdArray result = nn.Predict(imageArray)[0];
                    sw.Stop();

                    RILogManager.Default?.SendDebug("Result Time : " +
                      (sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L))).ToString("n0") + "μｓ");

                    int maxIndex = Array.IndexOf(result.Data, result.Data.Max());
                    RILogManager.Default?.SendDebug("[" + result.Data[maxIndex] + "] : " + classList[maxIndex]);
                } 
                while (ofd.ShowDialog() == DialogResult.OK);
            }
        }
    }
}
