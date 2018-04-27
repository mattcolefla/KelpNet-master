using System;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Activations
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) an elu. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.SingleInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class ELU : SingleInputFunction
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "ELU";

        /// <summary>   The alpha. </summary>
        private readonly Real _alpha;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Activations.ELU class.
        /// </summary>
        ///
        /// <param name="alpha">        (Optional) The alpha. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ELU(double alpha = 1, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            _alpha = alpha;

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
        private NdArray NeedPreviousForwardCpu([NotNull] NdArray x)
        {
            Real[] result = new Real[x.Data.Length];

            for (int i = 0; i < x.Data.Length; i++)
            {
                if (x.Data[i] >= 0)
                {
                    result[i] = x.Data[i];
                }
                else
                {
                    result[i] = _alpha * (Math.Exp(x.Data[i]) - 1);
                }
            }

            return NdArray.Convert(result, x.Shape, x.BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void NeedPreviousBackwardCpu([NotNull] NdArray y, [CanBeNull] NdArray x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                if (x.Data[i] >= 0)
                {
                    x.Grad[i] += y.Grad[i];
                }
                else
                {
                    x.Grad[i] += y.Grad[i] * _alpha * Math.Exp(x.Data[i]);
                }
            }
        }
    }
}
