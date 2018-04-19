using System;
using CIFARLoader;
using KelpNet.Common;

namespace KelpNetTester.TestData
{
    class CifarData
    {
        CIFARDataLoader cifarDataLoader = new CIFARDataLoader();

        private NdArray[] X;
        private NdArray[] Tx;

        private NdArray[] Y;
        private NdArray[] Ty;

        public CifarData()
        {
            //トレーニングデータ
            X = new NdArray[cifarDataLoader.TrainData.Length];
            //トレーニングデータラベル
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

            //教師データ
            Y = new NdArray[cifarDataLoader.TestData.Length];
            //教師データラベル
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

        public TestDataSet GetRandomYSet(int dataCount)
        {
            NdArray listY = new NdArray(new[] { 3, 32, 32 }, dataCount);
            NdArray listTy = new NdArray(new[] { 1 }, dataCount);

            for (int i = 0; i < dataCount; i++)
            {
                int index = Mother.Dice.Next(Y.Length);

                Array.Copy(Y[index].Data, 0, listY.Data, i * listY.Length, listY.Length);
                listTy.Data[i] = Ty[index].Data[0];
            }

            return new TestDataSet(listY, listTy);
        }

        public TestDataSet GetRandomXSet(int dataCount)
        {
            NdArray listX = new NdArray(new[] { 3, 32, 32 }, dataCount);
            NdArray listTx = new NdArray(new[] { 1 }, dataCount);

            for (int i = 0; i < dataCount; i++)
            {
                int index = Mother.Dice.Next(X.Length);

                Array.Copy(X[index].Data, 0, listX.Data, i * listX.Length, listX.Length);
                listTx.Data[i] = Tx[index].Data[0];
            }

            return new TestDataSet(listX, listTx);
        }
    }
}
