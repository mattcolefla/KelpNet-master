using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Mathmetrics.BasicMath
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   An add. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.DualInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class Add : DualInputFunction
    {
        /// <summary>   Name of the function. </summary>
        private const string FUNCTION_NAME = "Add";

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Mathmetrics.BasicMath.Add class.
        /// </summary>
        ///
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Add(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Forward CPU. </summary>
        ///
        /// <param name="a">    A NdArray to process. </param>
        /// <param name="b">    A NdArray to process. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected NdArray ForwardCpu(NdArray a, NdArray b)
        {
            Real[] resultData = new Real[a.Data.Length];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[i] + b.Data[i];
            }

            return new NdArray(resultData, a.Shape, a.BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="a">    A NdArray to process. </param>
        /// <param name="b">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected void BackwardCpu(NdArray y, NdArray a, NdArray b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                a.Grad[i] += y.Grad[i]; // * 1.0f
                b.Grad[i] += y.Grad[i]; // * 1.0f
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   An add constant. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.DualInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class AddConst : DualInputFunction
    {
        /// <summary>   Name of the function. </summary>
        private const string FUNCTION_NAME = "AddConst";

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Mathmetrics.BasicMath.AddConst class.
        /// </summary>
        ///
        /// <param name="name"> (Optional) The name. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public AddConst(string name = FUNCTION_NAME) : base(name)
        {
            DualInputForward = ForwardCpu;
            DualOutputBackward = BackwardCpu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Forward CPU. </summary>
        ///
        /// <param name="a">    A NdArray to process. </param>
        /// <param name="b">    A NdArray to process. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected NdArray ForwardCpu(NdArray a, NdArray b)
        {
            Real[] resultData = new Real[a.Data.Length];
            Real val = b.Data[0];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = a.Data[i] + val;
            }

            return new NdArray(resultData, a.Shape, a.BatchCount, this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="a">    A NdArray to process. </param>
        /// <param name="b">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected void BackwardCpu(NdArray y, NdArray a, NdArray b)
        {
            for (int i = 0; i < y.Grad.Length; i++)
            {
                a.Grad[i] += y.Grad[i]; // * 1.0f
            }
        }
    }

}
