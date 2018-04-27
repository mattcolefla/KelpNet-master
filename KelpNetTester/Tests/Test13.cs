using System;
using KelpNet.Common;
using KelpNet.Functions.Connections;
using KelpNet.Loss;
using KelpNet.Optimizers;

namespace KelpNetTester.Tests
{
    using ReflectSoftware.Insight;

    // Acquire a filter equivalent to the filter based on the image outputted by a certain learned filter
    // http://qiita.com/samacoba/items/958c02f455ca5f3a475d
    class Test13
    {
        public static void Run()
        {
            // Create a target filter (In case of practice, here is the unknown value)
            Deconvolution2D decon_core = new Deconvolution2D(1, 1, 15, 1, 7, gpuEnable: true)
            {
                Weight = { Data = MakeOneCore() }
            };

            Deconvolution2D model = new Deconvolution2D(1, 1, 15, 1, 7, gpuEnable: true);

            SGD optimizer = new SGD(learningRate: 0.00005); // diverge if big
            model.SetOptimizer(optimizer);
            MeanSquaredError meanSquaredError = new MeanSquaredError();

            // I am educating with the same educational image at the transplanting source, but changing to learning closer to practice
            for (int i = 0; i < 11; i++)
            {
                // Generate an image with randomly struck points
                NdArray img_p = getRandomImage();

                // Output a learning image with a target filter
                NdArray[] img_core = decon_core.Forward(img_p);

                // Output an image with an unlearned filter
                NdArray[] img_y = model.Forward(img_p);

                Real loss = meanSquaredError.Evaluate(img_y, img_core);

                model.Backward(img_y);
                model.Update();

                RILogManager.Default?.SendDebug("epoch" + i + " : " + loss);
            }
        }

        static NdArray getRandomImage(int N = 1, int img_w = 128, int img_h = 128)
        {
            // Make a 0.1% point randomly
            Real[] img_p = new Real[N * img_w * img_h];

            for (int i = 0; i < img_p.Length; i++)
            {
                img_p[i] = Mother.Dice.Next(0, 1000);
                img_p[i] = img_p[i] > 999 ? 0 : 1;
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
    }
}
