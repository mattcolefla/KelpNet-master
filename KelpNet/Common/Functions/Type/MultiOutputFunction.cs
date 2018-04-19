using System;

namespace KelpNet.Common.Functions.Type
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a multi output function. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Function"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public abstract class MultiOutputFunction : Function
    {
        /// <summary>   The single input forward. </summary>
        protected Func<NdArray, NdArray[]> SingleInputForward;
        /// <summary>   The single output backward. </summary>
        protected Action<NdArray[], NdArray> SingleOutputBackward;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Common.Functions.Type.MultiOutputFunction class.
        /// </summary>
        ///
        /// <param name="name">         The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected MultiOutputFunction(string name, string[] inputNames = null, string[] outputNames = null) : base(name, inputNames, outputNames)
        {
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Forwards the given xs. </summary>
        ///
        /// <param name="xs">   A variable-length parameters list containing xs. </param>
        ///
        /// <returns>   A NdArray[]. </returns>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.Function.Forward(params NdArray[])"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override NdArray[] Forward(params NdArray[] xs)
        {
            PrevInputs.Add(xs);

            xs[0].UseCount++;

            return SingleInputForward(xs[0]);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backwards the given ys. </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="ys">   A variable-length parameters list containing ys. </param>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.Function.Backward(params NdArray[])"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override void Backward(params NdArray[] ys)
        {
            NdArray[] xs = PrevInputs[PrevInputs.Count - 1];
            PrevInputs.RemoveAt(PrevInputs.Count - 1);

#if DEBUG
            if (xs == null || xs.Length != 1) throw new Exception("Invalid argument");
#endif
            BackwardCountUp();

            xs[0].UseCount--;

            SingleOutputBackward(ys, xs[0]);
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
            return SingleInputForward(xs[0]);
        }
    }
}
