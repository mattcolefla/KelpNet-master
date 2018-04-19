using System;
using KelpNet.Common.Functions.Type;

namespace KelpNet.Common.Functions.Container
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a split function. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.MultiOutputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class SplitFunction : MultiOutputFunction
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "SplitFunction";
        /// <summary>   The split number. </summary>
        private readonly int _splitNum;

        /// <summary>   The splited functions. </summary>
        public FunctionStack[] SplitedFunctions;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Common.Functions.Container.SplitFunction class.
        /// </summary>
        ///
        /// <param name="splitNum">     (Optional) The split number. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public SplitFunction(int splitNum = 2, string name = FUNCTION_NAME, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            _splitNum = splitNum;
            SplitedFunctions = new FunctionStack[splitNum];

            for (int i = 0; i < SplitedFunctions.Length; i++)
            {
                SplitedFunctions[i] = new FunctionStack(new Function[] { }, name + i, new[] { inputNames[0] }, new[] { outputNames[i] });
            }

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

        private NdArray[] ForwardCpu(NdArray x)
        {
            NdArray[] result = new NdArray[_splitNum];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = SplitedFunctions[i].Forward(x)[0];
            }

            return result;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward CPU. </summary>
        ///
        /// <param name="ys">   The ys. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void BackwardCpu(NdArray[] ys, NdArray x)
        {
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Evaluation function. </summary>
        ///
        /// <param name="xs">   A variable-length parameters list containing xs. </param>
        ///
        /// <returns>   A NdArray[]. </returns>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.Function.Predict(params NdArray[])"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override NdArray[] Predict(params NdArray[] xs)
        {
            NdArray[] result = new NdArray[_splitNum];

            for (int i = 0; i < result.Length; i++)
            {
                result[i] = SplitedFunctions[i].Predict(xs[0])[0];
            }

            return result;
        }
    }
}
