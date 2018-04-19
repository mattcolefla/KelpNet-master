using System;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Mathmetrics.Trigonometric
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   An arc cosine. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.SingleInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class ArcCos : SingleInputFunction
    {
        /// <summary>   Name of the function. </summary>
        private const string FUNCTION_NAME = "ArcCos";

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Mathmetrics.Trigonometric.ArcCos
        /// class.
        /// </summary>
        ///
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ArcCos(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Forward CPU. </summary>
        ///
        /// <param name="x">    A NdArray to process. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected NdArray ForwardCpu(NdArray x)
        {
            Real[] resultData = new Real[x.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = (Real)Math.Acos(x.Data[i]);
            }

            return new NdArray(resultData, x.Shape, x.BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected void BackwardCpu(NdArray y, NdArray x)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                x.Grad[i] += -1 / (Real)Math.Sqrt(-x.Data[i] * x.Data[i] + 1) * y.Grad[i];
            }
        }
    }
}
