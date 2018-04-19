using System;
using KelpNet.Common;
using MNISTLoader;

namespace KelpNetTester.TestData
{
    class MnistData
    {
        readonly MnistDataLoader mnistDataLoader = new MnistDataLoader();

        private NdArray[] X;
        private NdArray[] Tx;

        private NdArray[] Y;
        private NdArray[] Ty;

        public MnistData(int xdim)
        {
            X = new NdArray[mnistDataLoader.TrainData.Length];
            //Training data label
            Tx = new NdArray[mnistDataLoader.TrainData.Length];

            for (int i = 0; i < mnistDataLoader.TrainData.Length; i++)
            {
                Real[] x = new Real[xdim * xdim];
                for (int j = 0; j < mnistDataLoader.TrainData[i].Length; j++)
                {
                    x[j] = mnistDataLoader.TrainData[i][j] / 255.0;
                }
                X[i] = new NdArray(x, new[] { 1, xdim, xdim });

                Tx[i] = new NdArray(new[] { (Real)mnistDataLoader.TrainLabel[i] });
            }

            Y = new NdArray[mnistDataLoader.TeachData.Length];
            //Teacher data label
            Ty = new NdArray[mnistDataLoader.TeachData.Length];

            for (int i = 0; i < mnistDataLoader.TeachData.Length; i++)
            {
                Real[] y = new Real[xdim * xdim];
                for (int j = 0; j < mnistDataLoader.TeachData[i].Length; j++)
                {
                    y[j] = mnistDataLoader.TeachData[i][j] / 255.0;
                }
                Y[i] = new NdArray(y, new[] { 1, xdim, xdim });

                Ty[i] = new NdArray(new[] { (Real)mnistDataLoader.TeachLabel[i] });
            }
        }

        public TestDataSet GetRandomYSet(int dataCount, int xdim)
        {
            NdArray listY = new NdArray(new[] { 1, xdim, xdim }, dataCount);
            NdArray listTy = new NdArray(new[] { 1 }, dataCount);

            for (int i = 0; i < dataCount; i++)
            {
                int index = Mother.Dice.Next(Y.Length);

                Array.Copy(Y[index].Data, 0, listY.Data, i * listY.Length, listY.Length);
                listTy.Data[i] = Ty[index].Data[0];
            }

            return new TestDataSet(listY, listTy);
        }

        public TestDataSet GetRandomXSet(int dataCount, int x, int y)
        {
            NdArray listX = new NdArray(new[] { 1, x, y }, dataCount);
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
