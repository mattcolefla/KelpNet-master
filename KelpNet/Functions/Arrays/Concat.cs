using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Functions.Arrays
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   A concatenate. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.MultiInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class Concat : MultiInputFunction
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "Concat";
        /// <summary>   The axis. </summary>
        public int Axis;

        /// <summary>   The previous input sections. </summary>
        private readonly List<int[]> _prevInputSections = new List<int[]>();

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Arrays.Concat class.
        /// </summary>
        ///
        /// <param name="axis">         (Optional) The axis. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Concat(int axis = 1, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            Axis = axis;

            MultiInputForward = ForwardCpu;
            MultiOutputBackward = BackwardCpu;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Forward CPU. </summary>
        ///
        /// <param name="xs">   A variable-length parameters list containing xs. </param>
        ///
        /// <returns>   A NdArray. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        private NdArray ForwardCpu([NotNull] params NdArray[] xs)
        {
            int[] sections = new int[xs.Length - 1];
            int sizeOffset = xs[0].Shape[Axis];

            NdArray resultNdArray = xs[0].Clone();
            resultNdArray.ParentFunc = this;

            for (int i = 1; i < xs.Length; i++)
            {
                //It is logic that does not save the last shape because it is not used by Backward's Split
                sections[i - 1] = sizeOffset;
                sizeOffset += xs[i].Shape[Axis];

                resultNdArray = NdArray.Concatenate(resultNdArray, xs[i], Axis);
            }

            _prevInputSections.Add(sections);

            return resultNdArray;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="xs">   The xs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void BackwardCpu([NotNull] NdArray y, [NotNull] NdArray[] xs)
        {
            int[] prevInputShapes = _prevInputSections[_prevInputSections.Count - 1];
            _prevInputSections.RemoveAt(_prevInputSections.Count - 1);

            NdArray[] result = NdArray.Split(y, prevInputShapes, Axis);

            for (int i = 0; i < xs.Length; i++)
            {
                for (int j = 0; j < xs[i].Grad.Length; j++)
                {
                    xs[i].Grad[j] += result[i].Grad[j];
                }
            }
        }
    }
}
