using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Arrays
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   A split axis. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.MultiOutputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class SplitAxis : MultiOutputFunction
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "SplitAxis";
        /// <summary>   The axis. </summary>
        public int Axis;
        /// <summary>   The indices. </summary>
        public int[] Indices;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Arrays.SplitAxis class.
        /// </summary>
        ///
        /// <param name="indices">      The indices. </param>
        /// <param name="axis">         The axis. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public SplitAxis(int indices, int axis, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            Indices = new[] { indices };
            Axis = axis;

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Arrays.SplitAxis class.
        /// </summary>
        ///
        /// <param name="indices">      The indices. </param>
        /// <param name="axis">         The axis. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public SplitAxis([NotNull] int[] indices, int axis, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            Indices = indices.ToArray();
            Axis = axis;

            SingleInputForward = ForwardCpu;
            SingleOutputBackward = BackwardCpu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Forward CPU. </summary>
        ///
        /// <param name="x">    A NdArray to process. </param>
        ///
        /// <returns>   A NdArray[]. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        private NdArray[] ForwardCpu([NotNull] NdArray x)
        {
            NdArray[] resultArays = NdArray.Split(x, Indices, Axis);

            foreach (var resultArray in resultArays)
            {
                resultArray.ParentFunc = this;
            }

            return resultArays;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward CPU. </summary>
        ///
        /// <param name="ys">   The ys. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void BackwardCpu([NotNull] NdArray[] ys, [NotNull] NdArray x)
        {
            NdArray resultNdArray = ys[0].Clone();

            for (int i = 1; i < ys.Length; i++)
            {
                resultNdArray = NdArray.Concatenate(resultNdArray, ys[i], Axis);
            }

            for (int i = 0; i < x.Grad.Length; i++)
            {
                x.Grad[i] += resultNdArray.Grad[i];
            }
        }
    }
}
