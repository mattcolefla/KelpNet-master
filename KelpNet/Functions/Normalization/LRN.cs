using System;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Normalization
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   A lrn. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.SingleInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class LRN : SingleInputFunction
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "LRN";

        /// <summary>   An int to process. </summary>
        private int n;
        /// <summary>   A Real to process. </summary>
        private Real k;
        /// <summary>   The alpha. </summary>
        private Real alpha;
        /// <summary>   The beta. </summary>
        private Real beta;
        /// <summary>   The unit scale. </summary>
        private Real[] unitScale;
        /// <summary>   The scale. </summary>
        private Real[] scale;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Normalization.LRN class.
        /// </summary>
        ///
        /// <param name="n">            (Optional) An int to process. </param>
        /// <param name="k">            (Optional) A double to process. </param>
        /// <param name="alpha">        (Optional) The alpha. </param>
        /// <param name="beta">         (Optional) The beta. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public LRN(int n = 5, double k = 2, double alpha = 1e-4, double beta = 0.75, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            this.n = n;
            this.k = (Real)k;
            this.alpha = (Real)alpha;
            this.beta = (Real)beta;

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
        private NdArray NeedPreviousForwardCpu([NotNull] NdArray input)
        {
            int nHalf = n / 2;
            Real[] result = new Real[input.Data.Length];
            Real[] x2 = new Real[input.Data.Length];
            Real[] sumPart = new Real[input.Data.Length];
            unitScale = new Real[input.Data.Length];
            scale = new Real[input.Data.Length];

            for (int i = 0; i < x2.Length; i++)
            {
                x2[i] = input.Data[i] * input.Data[i];
            }
            Array.Copy(x2, sumPart, x2.Length);


            for (int b = 0; b < input.BatchCount; b++)
            {
                for (int ich = 0; ich < input.Shape[0]; ich++)
                {
                    for (int location = 0; location < input.Shape[1] * input.Shape[2]; location++)
                    {
                        int baseIndex = b * input.Length + ich * input.Shape[1] * input.Shape[2] + location;

                        for (int offsetCh = 1; offsetCh < nHalf; offsetCh++)
                        {
                            if (ich - offsetCh > 0)
                            {
                                int offsetIndex = b * input.Length + (ich - offsetCh) * input.Shape[1] * input.Shape[2] + location;
                                sumPart[baseIndex] += x2[offsetIndex];
                            }

                            if (ich + offsetCh < input.Shape[0])
                            {
                                int offsetIndex = b * input.Length + (ich + offsetCh) * input.Shape[1] * input.Shape[2] + location;
                                sumPart[baseIndex] += x2[offsetIndex];
                            }
                        }
                    }
                }
            }

            // Take the average of places with n channels before and after
            for (int i = 0; i < sumPart.Length; i++)
            {
                unitScale[i] = k + alpha * sumPart[i];
                scale[i] = Math.Pow(unitScale[i], -beta);
                result[i] *= scale[i];
            }

            return NdArray.Convert(result, input.Shape, input.BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void NeedPreviousBackwardCpu([NotNull] NdArray y, [NotNull] NdArray x)
        {
            int nHalf = n / 2;
            Real[] summand = new Real[y.Grad.Length];
            Real[] sumPart = new Real[y.Grad.Length];

            for (int i = 0; i < y.Grad.Length; i++)
            {
                summand[i] = y.Data[i] * y.Grad[i] / unitScale[i];
            }

            Array.Copy(summand, sumPart, summand.Length);

            for (int b = 0; b < y.BatchCount; b++)
            {
                for (int ich = 0; ich < y.Shape[0]; ich++)
                {
                    for (int location = 0; location < y.Shape[1] * y.Shape[2]; location++)
                    {
                        int baseIndex = b * y.Length + ich * y.Shape[1] * y.Shape[2] + location;

                        for (int offsetCh = 1; offsetCh < nHalf; offsetCh++)
                        {
                            if (ich - offsetCh > 0)
                            {
                                int offsetIndex = b * y.Length + (ich - offsetCh) * y.Shape[1] * y.Shape[2] + location;
                                sumPart[baseIndex] += summand[offsetIndex];
                            }

                            if (ich + offsetCh < y.Shape[0])
                            {
                                int offsetIndex = b * y.Length + (ich + offsetCh) * y.Shape[1] * y.Shape[2] + location;
                                sumPart[baseIndex] += summand[offsetIndex];
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += y.Grad[i] * scale[i] - 2 * alpha * beta * y.Data[i] * sumPart[i];
            }
        }
    }
}
