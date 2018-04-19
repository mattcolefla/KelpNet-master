using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Arrays
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   A reshape. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.SingleInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    class Reshape : SingleInputFunction
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "Reshape";
        /// <summary>   The shape. </summary>
        public int[] Shape;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Arrays.Reshape class.
        /// </summary>
        ///
        /// <param name="shape">        The shape. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Reshape(int[] shape, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            Shape = shape;

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Forward CPU. </summary>
        ///
        /// <param name="val">  The value. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        NdArray ForwardCpu(NdArray val)
        {
            NdArray result = val.Clone();
            result.ParentFunc = this;
            result.Reshape(Shape);

            return result;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        void BackwardCpu(NdArray y, NdArray x)
        {
            y.Grad = x.Grad.ToArray();
        }
    }
    
}
