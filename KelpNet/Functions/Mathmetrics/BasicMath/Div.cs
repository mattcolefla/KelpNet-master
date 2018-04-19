using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Mathmetrics.BasicMath
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   A div. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.DualInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class Div : DualInputFunction
    {
        /// <summary>   Name of the function. </summary>
        private const string FUNCTION_NAME = "Div";

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Mathmetrics.BasicMath.Div class.
        /// </summary>
        ///
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Div(string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
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
                resultData[i] = a.Data[i] / b.Data[i];
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
                Real gx = y.Grad[i] / b.Data[i];
                a.Grad[i] += gx;
                b.Grad[i] += -gx * a.Data[i] / b.Data[i];
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Constant on the right side. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.DualInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class DivConst : DualInputFunction
    {
        /// <summary>   Name of the function. </summary>
        private const string FUNCTION_NAME = "DivConst";

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Mathmetrics.BasicMath.DivConst class.
        /// </summary>
        ///
        /// <param name="name"> (Optional) The name. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public DivConst(string name = FUNCTION_NAME) : base(name)
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
                resultData[i] = a.Data[i] / val;
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
            Real val = b.Data[0];

            for (int i = 0; i < y.Grad.Length; i++)
            {
                a.Grad[i] += y.Grad[i] / val;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Constant on the left side. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.DualInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class ConstDiv : DualInputFunction
    {
        /// <summary>   Name of the function. </summary>
        private const string FUNCTION_NAME = "ConstDiv";

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Mathmetrics.BasicMath.ConstDiv class.
        /// </summary>
        ///
        /// <param name="name"> (Optional) The name. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ConstDiv(string name = FUNCTION_NAME) : base(name)
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
            Real val = a.Data[0];

            for (int i = 0; i < resultData.Length; i++)
            {
                resultData[i] = val / b.Data[i];
            }

            return new NdArray(resultData, b.Shape, b.BatchCount, this);
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
            Real val = a.Data[0];

            for (int i = 0; i < y.Grad.Length; i++)
            {
                Real gx = y.Grad[i] / b.Data[i];
                b.Grad[i] += -gx * val / b.Data[i];
            }
        }
    }

}
