using System;
using System.Drawing;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Poolings
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) an average pooling. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.SingleInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class AveragePooling : SingleInputFunction
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "AveragePooling";

        /// <summary>   The height. </summary>
        private int _kHeight;
        /// <summary>   The width. </summary>
        private int _kWidth;
        /// <summary>   The pad y coordinate. </summary>
        private int _padY;
        /// <summary>   The pad x coordinate. </summary>
        private int _padX;
        /// <summary>   The stride x coordinate. </summary>
        private int _strideX;
        /// <summary>   The stride y coordinate. </summary>
        private int _strideY;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Poolings.AveragePooling class.
        /// </summary>
        ///
        /// <param name="ksize">        The ksize. </param>
        /// <param name="stride">       (Optional) The stride. </param>
        /// <param name="pad">          (Optional) The pad. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public AveragePooling(int ksize, int stride = 1, int pad = 0, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            _kWidth = ksize;
            _kHeight = ksize;
            _padY = pad;
            _padX = pad;
            _strideX = stride;
            _strideY = stride;

            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Poolings.AveragePooling class.
        /// </summary>
        ///
        /// <param name="ksize">        The ksize. </param>
        /// <param name="stride">       (Optional) The stride. </param>
        /// <param name="pad">          (Optional) The pad. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public AveragePooling(Size ksize, Size stride = new Size(), Size pad = new Size(), [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            if (pad == Size.Empty)
                pad = new Size(0, 0);

            if (stride == Size.Empty)
                stride = new Size(1, 1);

            _kWidth = ksize.Width;
            _kHeight = ksize.Height;
            _padY = pad.Height;
            _padX = pad.Width;
            _strideX = stride.Width;
            _strideY = stride.Height;

            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous forward CPU. </summary>
        ///
        /// <param name="input">    The input. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        protected NdArray NeedPreviousForwardCpu([NotNull] NdArray input)
        {
            int outputHeight = (int)Math.Floor((input.Shape[1] - _kHeight + _padY * 2.0) / _strideY) + 1;
            int outputWidth = (int)Math.Floor((input.Shape[2] - _kWidth + _padX * 2.0) / _strideX) + 1;
            Real[] result = new Real[input.Shape[0] * outputHeight * outputWidth * input.BatchCount];
            Real m = _kHeight * _kWidth;

            for (int b = 0; b < input.BatchCount; b++)
            {
                int resultIndex = b * input.Shape[0] * outputHeight * outputWidth;

                for (int i = 0; i < input.Shape[0]; i++)
                {
                    int inputIndexOffset = i * input.Shape[1] * input.Shape[2];

                    for (int y = 0; y < outputHeight; y++)
                    {
                        int dyOffset = y * _strideY - _padY < 0 ? 0 : y * _strideY - _padY;
                        int dyLimit = _kHeight + dyOffset < input.Shape[1] ? _kHeight + dyOffset : input.Shape[1];

                        for (int x = 0; x < outputWidth; x++)
                        {
                            int dxOffset = x * _strideX - _padX < 0 ? 0 : x * _strideX - _padX;
                            int dxLimit = _kWidth + dxOffset < input.Shape[2] ? _kWidth + dxOffset : input.Shape[2];

                            for (int dy = dyOffset; dy < dyLimit; dy++)
                            {
                                for (int dx = dxOffset; dx < dxLimit; dx++)
                                {
                                    int inputindex = inputIndexOffset + dy * input.Shape[2] + dx;
                                    result[resultIndex] += input.Data[inputindex + input.Length * b] / m;
                                }
                            }

                            resultIndex++;
                        }
                    }
                }
            }

            return NdArray.Convert(result, new[] { input.Shape[0], outputHeight, outputWidth }, input.BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected void NeedPreviousBackwardCpu([NotNull] NdArray y, [CanBeNull] NdArray x)
        {
            Real m = _kHeight * _kWidth;

            for (int b = 0; b < y.BatchCount; b++)
            {
                int gyIndex = b * y.Length;

                for (int i = 0; i < x.Shape[0]; i++)
                {
                    int resultIndexOffset = b * x.Length + i * x.Shape[1] * x.Shape[2];

                    for (int posY = 0; posY < y.Shape[1]; posY++)
                    {
                        int dyOffset = posY * _strideY - _padY < 0 ? 0 : posY * _strideY - _padY;
                        int dyLimit = _kHeight + dyOffset < x.Shape[1] ? _kHeight + dyOffset : x.Shape[1];

                        for (int posX = 0; posX < y.Shape[2]; posX++)
                        {
                            int dxOffset = posX * _strideX - _padX < 0 ? 0 : posX * _strideX - _padX;
                            int dxLimit = _kWidth + dxOffset < x.Shape[2] ? _kWidth + dxOffset : x.Shape[2];

                            Real gyData = y.Grad[gyIndex] / m;

                            for (int dy = dyOffset; dy < dyLimit; dy++)
                            {
                                for (int dx = dxOffset; dx < dxLimit; dx++)
                                {
                                    int resultIndex = resultIndexOffset + dy * x.Shape[2] + dx;
                                    x.Grad[resultIndex] += gyData;
                                }
                            }

                            gyIndex++;
                        }
                    }
                }
            }
        }
    }
}
