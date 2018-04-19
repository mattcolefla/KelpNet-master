using System;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Activations
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   A softplus. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.SingleInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class Softplus : SingleInputFunction
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "Softplus";

        /// <summary>   The beta. </summary>
        private readonly Real _beta;
        /// <summary>   The beta inverse. </summary>
        private readonly Real _betaInv;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Activations.Softplus class.
        /// </summary>
        ///
        /// <param name="beta">         (Optional) The beta. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Softplus(double beta = 1, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            _beta = beta;
            _betaInv = 1 / _beta;

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

        protected NdArray NeedPreviousForwardCpu(NdArray x)
        {
            Real[] y = new Real[x.Data.Length];

            for (int b = 0; b < x.BatchCount; b++)
            {
                for (int i = 0; i < x.Length; i++)
                {
                    y[i + b * x.Length] = x.Data[i + b * x.Length] * _beta;
                }

                Real maxval = y[b * x.Length];
                for (int i = 1; i < x.Length; i++)
                {
                    if (maxval < y[i + b * x.Length])
                    {
                        maxval = y[i + b * x.Length];
                    }
                }

                for (int i = 0; i < x.Length; i++)
                {
                    y[i + b * x.Length] = (maxval + Math.Log(1.0 + Math.Exp(-Math.Abs(x.Data[i + b * x.Length] * _beta)))) * _betaInv;
                }

            }

            return NdArray.Convert(y, x.Shape, x.BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Need previous backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected void NeedPreviousBackwardCpu(NdArray y, NdArray x)
        {
            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += (1 - 1 / (1 + Math.Exp(_beta * y.Data[i]))) * y.Grad[i];
            }

        }
    }
}

