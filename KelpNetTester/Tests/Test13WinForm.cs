using System;
using System.Drawing;
using System.Windows.Forms;
using KelpNet.Common;
using KelpNet.Common.Tools;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    public partial class Test13WinForm : Form
    {
        readonly Deconvolution2D model;
        private readonly Deconvolution2D decon_core;
        private readonly SGD optimizer;
        readonly MeanSquaredError meanSquaredError = new MeanSquaredError();
        private int counter = 0;

        public Test13WinForm()
        {
            InitializeComponent();

            ClientSize = new Size(128 * 4, 128 * 4);

            // Create a target filter (In case of practice, here is the unknown value)
            decon_core = new Deconvolution2D(true, 1, 1, 15, 1, 7, gpuEnable: true)
            {
                Weight = { Data = MakeOneCore() }
            };

            model = new Deconvolution2D(true, 1, 1, 15, 1, 7, gpuEnable: true);

            optimizer = new SGD(learningRate: 0.01); // diverge if big
            model.SetOptimizer(optimizer);
        }

        static NdArray getRandomImage(int N = 1, int img_w = 128, int img_h = 128)
        {
            // Make a 0.1% point randomly
            Real[] img_p = new Real[N * img_w * img_h];

            for (int i = 0; i < img_p.Length; i++)
            {
                img_p[i] = Mother.Dice.Next(0, 10000);
                img_p[i] = img_p[i] < 10 ? 255 : 0;
            }

            return new NdArray(img_p, new[] { N, img_h, img_w }, 1);
        }

        // Create one spherical pattern (Gauss)
        static Real[] MakeOneCore()
        {
            int max_xy = 15;
            Real sig = 5;
            Real sig2 = sig * sig;
            Real c_xy = 7;
            Real[] core = new Real[max_xy * max_xy];

            for (int px = 0; px < max_xy; px++)
            {
                for (int py = 0; py < max_xy; py++)
                {
                    Real r2 = (px - c_xy) * (px - c_xy) + (py - c_xy) * (py - c_xy);
                    core[py * max_xy + px] = Math.Exp(-r2 / sig2) * 1;
                }
            }

            return core;
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            // I am educating with the same educational image at the transplanting source, but changing to learning closer to practice
            if (counter < 11)
            {
                // Generate an image with randomly struck points
                NdArray img_p = getRandomImage();

                // Output a learning image with a target filter
                NdArray[] img_core = decon_core?.Forward(true, img_p);

                // Output an image with an unlearned filter
                NdArray[] img_y = model?.Forward(true, img_p);

                // implicitly use img_y as NdArray
                BackgroundImage = NdArrayConverter.NdArray2Image(img_y[0].GetSingleArray(0));

                Real loss = meanSquaredError.Evaluate(img_y, img_core);

                model.Backward(true, img_y);
                model.Update();

                Text = "[epoch" + counter + "] Loss : " + $"{loss:F4}";

                counter++;
            }
            else
            {
                timer1.Enabled = false;
            }

        }
    }
}
