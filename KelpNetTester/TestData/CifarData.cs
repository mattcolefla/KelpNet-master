using System;
using CIFARLoader;
using KelpNet.Common;
using KelpNet.Common.Functions;
using ReflectSoftware.Insight;

namespace KelpNetTester.TestData
{
    /// <summary>   A cifar data. </summary>
    class CifarData
    {
        /// <summary>   The cifar data loader. </summary>
        CIFARDataLoader cifarDataLoader = new CIFARDataLoader();

        /// <summary>   A NdArray[] to process. </summary>
        private NdArray[] X;
        /// <summary>   The transmit. </summary>
        private NdArray[] Tx;

        /// <summary>   A NdArray[] to process. </summary>
        private NdArray[] Y;
        /// <summary>   The ty. </summary>
        private NdArray[] Ty;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNetTester.TestData.CifarData class.
        /// </summary>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public CifarData()
        {
            // training data
            X = new NdArray[cifarDataLoader.TrainData.Length];
            // training data label
            Tx = new NdArray[cifarDataLoader.TrainData.Length];

            for (int i = 0; i < cifarDataLoader.TrainData.Length; i++)
            {
                Real[] x = new Real[3 * 32 * 32];
                for (int j = 0; j < cifarDataLoader.TrainData[i].Length; j++)
                {
                    x[j] = cifarDataLoader.TrainData[i][j] / 255.0;
                }
                X[i] = new NdArray(x, new[] { 3, 32, 32 });

                Tx[i] = new NdArray(new[] { (Real)cifarDataLoader.TrainLabel[i] });
            }

            // Teacher data
            Y = new NdArray[cifarDataLoader.TestData.Length];
            // teacher data label
            Ty = new NdArray[cifarDataLoader.TestData.Length];

            for (int i = 0; i < cifarDataLoader.TestData.Length; i++)
            {
                Real[] y = new Real[3 * 32 * 32];

                for (int j = 0; j < cifarDataLoader.TestData[i].Length; j++)
                {
                    y[j] = cifarDataLoader.TestData[i][j] / 255.0;
                }

                Y[i] = new NdArray(y, new[] { 3, 32, 32 });

                Ty[i] = new NdArray(new[] { (Real)cifarDataLoader.TestLabel[i] });
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets random y coordinate set. </summary>
        ///
        /// <param name="dataCount">    Number of data. </param>
        ///
        /// <returns>   The random y coordinate set. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public TestDataSet GetRandomYSet(int dataCount)
        {
            NdArray listY = new NdArray(new[] { 3, 32, 32 }, dataCount, (Function)null);
            NdArray listTy = new NdArray(new[] { 1 }, dataCount, (Function)null);

            for (int i = 0; i < dataCount; i++)
            {
                int index = Mother.Dice.Next(Y.Length);

                Array.Copy(Y[index].Data, 0, listY.Data, i * listY.Length, listY.Length);
                listTy.Data[i] = Ty[index].Data[0];
            }

            TestDataSet tds = new TestDataSet(listY, listTy);
            RILogManager.Default?.SendDebug("Getting random Y data (" + tds.Data.Length + ") bytes");
            return tds;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets random x coordinate set. </summary>
        ///
        /// <param name="dataCount">    Number of data. </param>
        ///
        /// <returns>   The random x coordinate set. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public TestDataSet GetRandomXSet(int dataCount)
        {
            NdArray listX = new NdArray(new[] { 3, 32, 32 }, dataCount, (Function)null);
            NdArray listTx = new NdArray(new[] { 1 }, dataCount, (Function)null);

            for (int i = 0; i < dataCount; i++)
            {
                int index = Mother.Dice.Next(X.Length);

                Array.Copy(X[index].Data, 0, listX.Data, i * listX.Length, listX.Length);
                listTx.Data[i] = Tx[index].Data[0];
            }

            TestDataSet tds = new TestDataSet(listX, listTx);
            RILogManager.Default?.SendDebug("Getting random X data (" + tds.Data.Length + ") bytes");
            return tds;
        }
    }
}
