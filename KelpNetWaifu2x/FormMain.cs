﻿using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using KelpNet.Common;
using KelpNet.Common.Functions;
using KelpNet.Common.Functions.Container;
using KelpNet.Common.Tools;
using KelpNet.Functions.Activations;
using KelpNet.Functions.Connections;

namespace KelpNetWaifu2x
{
    /* モデルファイルを https://github.com/nagadomi/waifu2x/tree/master/models/upconv_7/art よりダウンロードしてください*/
    /* The sample is confirmed to work with scale 2.0 x _ modelel.json*/

    public partial class FormMain : Form
    {
        FunctionStack nn;

        public FormMain()
        {
            InitializeComponent();

            //GPUを初期化
            Weaver.Initialize(ComputeDeviceTypes.Gpu);
        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog
            {
                Filter = "JsonFiles(*.json)|*.json|All Files(*.*)|*.*",
            };

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                int layerCounter = 1;

                var json = DynamicJson.Parse(File.ReadAllText(ofd.FileName));

                List<Function> functionList = new List<Function>();

                //Microsoft.CSharp.RuntimeBinder.RuntimeBinderExceptionは無視して下さい
                foreach (var data in json)
                {
                    Real[,,,] weightData = new Real[(int)data["nOutputPlane"], (int)data["nInputPlane"], (int)data["kW"], (int)data["kH"]];

                    for (int i = 0; i < weightData.GetLength(0); i++)
                    {
                        for (int j = 0; j < weightData.GetLength(1); j++)
                        {
                            for (int k = 0; k < weightData.GetLength(2); k++)
                            {
                                for (int l = 0; l < weightData.GetLength(3); l++)
                                {
                                    weightData[i, j, k, l] = data["weight"][i][j][k][l];
                                }
                            }
                        }
                    }

                    // pad to fit size of input and output image
                    functionList.Add(new Convolution2D((int)data["nInputPlane"], (int)data["nOutputPlane"], (int)data["kW"], pad: (int)data["kW"] / 2, initialW: weightData, initialb: (Real[])data["bias"],name: "Convolution2D l" + layerCounter++, gpuEnable: true));
                    functionList.Add(new LeakyReLU(0.1, name: "LeakyReLU l" + layerCounter++));
                }

                nn = new FunctionStack(functionList.ToArray());
                nn.Compress();

                MessageBox.Show("Load completed");
            }
        }

        Bitmap _baseImage;
        private void button2_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog
            {
                Filter = "Image Files(*.jpg;*.png)|*.jpg;*.png|All Files(*.*)|*.*"
            };

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                _baseImage = new Bitmap(ofd.FileName);
                pictureBox1.Image = new Bitmap(_baseImage);
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            SaveFileDialog sfd = new SaveFileDialog
            {
                Filter = "png Files(*.png)|*.png|All Files(*.*)|*.*",
                FileName = "result.png"
            };

            if (sfd.ShowDialog() == DialogResult.OK)
            {
                Task.Factory.StartNew(() =>
                {
                    // We need to enlarge in advance before entering the network
                    Bitmap resultImage = new Bitmap(_baseImage.Width * 2, _baseImage.Height * 2, PixelFormat.Format24bppRgb);
                    Graphics g = Graphics.FromImage(resultImage);

                    // use nearest neighbor for interpolation
                    g.InterpolationMode = InterpolationMode.NearestNeighbor;

                    // Enlarge and draw the image
                    g.DrawImage(_baseImage, 0, 0, _baseImage.Width * 2, _baseImage.Height * 2);
                    g.Dispose();

                    NdArray image = NdArrayConverter.Image2NdArray(resultImage);
                    NdArray[] resultArray = nn.Predict(image);
                    resultImage = NdArrayConverter.NdArray2Image(resultArray[0].GetSingleArray(0));
                    resultImage.Save(sfd.FileName);
                    pictureBox1.Image = new Bitmap(resultImage);
                }
                    ).ContinueWith(_ =>
                    {
                        MessageBox.Show("After the exchange finished");
                    });

                MessageBox.Show("The conversion process has started. \n Please wait for a while until \"conversion complete\" message is displayed \n * It will take a very long time (about 3 minutes with 64 x 64 images)");
            }
        }
    }
}
