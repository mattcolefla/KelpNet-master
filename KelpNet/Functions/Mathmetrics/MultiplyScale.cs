using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet.Common;
using KelpNet.Common.Functions.Type;
using KelpNet.Functions.Arrays;

namespace KelpNet.Functions.Mathmetrics
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <inheritdoc />
    ///  <summary>   (Serializable) a multiply scale. </summary>
    ///  <seealso cref="T:KelpNet.Common.Functions.Type.SingleInputFunction" />

    [Serializable]
    public sealed class MultiplyScale : SingleInputFunction
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "MultiplyScale";

        /// <summary>   The axis. </summary>
        private int Axis;
        /// <summary>   The weight. </summary>
        public NdArray Weight;
        /// <summary>   The bias. </summary>
        public NdArray Bias;
        /// <summary>   True to bias term. </summary>
        public bool BiasTerm = false;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Mathmetrics.MultiplyScale class.
        /// </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="axis">         (Optional) The axis. </param>
        /// <param name="wShape">       (Optional) The shape. </param>
        /// <param name="biasTerm">     (Optional) True to bias term. </param>
        /// <param name="initialW">     (Optional) The initial w. </param>
        /// <param name="initialb">     (Optional) The initialb. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public MultiplyScale(int axis = 1, [NotNull] int[] wShape = null, bool biasTerm = false, [CanBeNull] Array initialW = null, [CanBeNull] Array initialb = null, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            Axis = axis;
            BiasTerm = biasTerm;

#if DEBUG
            if (wShape == null)
            {
                if (biasTerm)
                {
                    throw new Exception("Please use AddBias for use only with Bias");
                }

                throw new Exception("Parameter setting is incorrect");
            }
#endif
            Weight = new NdArray(wShape);
            Parameters = new NdArray[biasTerm ? 2 : 1];

            Weight.Data = initialW == null ? Enumerable.Repeat((Real) 1.0, Weight.Data.Length).ToArray() : Real.GetArray(initialW);

            Parameters[0] = Weight;

            if (biasTerm)
            {
                Bias = new NdArray(wShape);

                if (initialb != null)
                {
                    Bias.Data = Real.GetArray(initialb);
                }

                Parameters[1] = Bias;
            }

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
        private NdArray ForwardCpu([NotNull] NdArray x)
        {
            int[] inputShape = x.Shape;
            int[] outputShape = Weight.Shape;

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

            int[] preShape = shapeList.ToArray();

            NdArray y1 = new Reshape(preShape).Forward(Weight)[0];
            NdArray y2 = new Broadcast(inputShape).Forward(y1)[0];

            if (BiasTerm)
            {
                NdArray b1 = new Reshape(preShape).Forward(Bias)[0];
                NdArray b2 = new Broadcast(inputShape).Forward(b1)[0];

                return x * y2 + b2;
            }

            return x * y2;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward CPU. </summary>
        ///
        /// <param name="y">    A NdArray to process. </param>
        /// <param name="x">    A NdArray to process. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        private void BackwardCpu([CanBeNull] NdArray y, [CanBeNull] NdArray x)
        {
        }
    }
}
