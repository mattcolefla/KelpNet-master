using System;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Activations
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a softmax. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.SingleInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class Softmax : SingleInputFunction
    {
        /// <summary>   Name of the function. </summary>
        private const string FUNCTION_NAME = "Softmax";

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Activations.Softmax class.
        /// </summary>
        ///
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Softmax([CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            SingleInputForward = NeedPreviousForwardCpu;
            SingleOutputBackward = NeedPreviousBackwardCpu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous forward CPU. </summary>
        ///
        /// <param name="x">    A NdArray to process. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        protected NdArray NeedPreviousForwardCpu([NotNull] NdArray x)
        {
            Real[] y = new Real[x.Data.Length];

            int indexOffset = 0;

            for (int b = 0; b < x.BatchCount; b++)
            {
                Real maxval = x.Data[indexOffset];

                for (int i = 1; i < x.Length; i++)
                {
                    if (maxval < x.Data[indexOffset + i])
                        maxval = x.Data[indexOffset + i];
                }

                Real sumval = 0;

                for (int i = 0; i < x.Length; i++)
                {
                    y[indexOffset + i] = Math.Exp(x.Data[indexOffset + i] - maxval);
                    sumval += y[indexOffset + i];
                }

                for (int i = 0; i < x.Length; i++)
                    y[indexOffset + i] /= sumval;

                indexOffset += x.Length;
            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected void NeedPreviousBackwardCpu([NotNull] NdArray y, [NotNull] NdArray x)
        {
            Real[] gx = new Real[y.Grad.Length];

            int indexOffset = 0;

            for (int b = 0; b < y.BatchCount; b++)
            {
                Real sumdx = 0;

                for (int i = 0; i < y.Length; i++)
                {
                    gx[indexOffset + i] = y.Data[indexOffset + i] * y.Data[indexOffset + i];
                    sumdx += gx[indexOffset + i];
                }

                for (int i = 0; i < y.Length; i++)
                    gx[indexOffset + i] -= y.Data[indexOffset + i] * sumdx;

                indexOffset += y.Length;
            }

            for (int i = 0; i < x.Grad.Length; i++)
                x.Grad[i] += gx[i];
        }
    }
}
