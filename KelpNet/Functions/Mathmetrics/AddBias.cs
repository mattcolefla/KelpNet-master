using System;
using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;
using KelpNet.Functions.Arrays;

namespace KelpNet.Functions.Mathmetrics
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) an add bias. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Type.SingleInputFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class AddBias : SingleInputFunction
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "AddBias";

        /// <summary>   The axis. </summary>
        private int Axis;
        /// <summary>   The bias. </summary>
        private NdArray Bias;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Mathmetrics.AddBias class.
        /// </summary>
        ///
        /// <param name="axis">         (Optional) The axis. </param>
        /// <param name="biasShape">    (Optional) The bias shape. </param>
        /// <param name="initialb">     (Optional) The initialb. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public AddBias(int axis = 1, [NotNull] int[] biasShape = null, [CanBeNull] Array initialb = null, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            Axis = axis;
            Bias = new NdArray(biasShape);

            if (initialb != null)
            {
                Bias.Data = Real.GetArray(initialb);
            }

            Parameters = new[] { Bias };

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

        [CanBeNull]
        protected NdArray ForwardCpu([NotNull] NdArray x)
        {
            int[] inputShape = x.Shape;
            int[] outputShape = Bias.Shape;

            List<int> shapeList = new List<int>();

            for (int i = 0; i < Axis; i++)
            {
                shapeList.Add(1);
            }

            shapeList.AddRange(outputShape);

            for (int i = 0; i < inputShape.Length - Axis - outputShape.Length; i++)
            {
                shapeList.Add(1);
            }

            int[] y1Shape = shapeList.ToArray();

            NdArray y1 = new Reshape(y1Shape).Forward(Bias)[0];
            NdArray y2 = new Broadcast(inputShape).Forward(y1)[0];

            return x + y2;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected void BackwardCpu([CanBeNull] NdArray y, [CanBeNull] NdArray x)
        {
        }
    }
}
