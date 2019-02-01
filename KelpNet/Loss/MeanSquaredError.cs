using System;
using KelpNet.Common;
using KelpNet.Common.Loss;

namespace KelpNet.Loss
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   A mean squared error. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Loss.LossFunction"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class MeanSquaredError : LossFunction
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Evaluates. </summary>
        ///
        /// <exception cref="Exception">    Thrown when an exception error condition occurs. </exception>
        ///
        /// <param name="input">        The input. </param>
        /// <param name="teachSignal">  The teach signal. </param>
        ///
        /// <returns>   A Real. </returns>
        ///
        /// <seealso cref="M:KelpNet.Common.Loss.LossFunction.Evaluate(NdArray[],NdArray[])"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override Real Evaluate([NotNull] NdArray[] input, [NotNull] NdArray[] teachSignal)
        {
            Real resultLoss = 0;

#if DEBUG
            if (input.Length != teachSignal.Length) throw new Exception("Input and teacher signal size are different");
#endif

            for (int k = 0; k < input.Length; k++)
            {
                Real sumLoss = 0;
                Real[] resultArray = new Real[input[k].Data.Length];

                for (int b = 0; b < input[k].BatchCount; b++)
                {
                    Real localloss = 0;
                    Real coeff = 2.0 / teachSignal[k].Length;
                    int batchoffset = b * teachSignal[k].Length;

                    for (int i = 0; i < input[k].Length; i++)
                    {
                        Real result = input[k].Data[b * input[k].Length + i] - teachSignal[k].Data[batchoffset + i];
                        localloss += result * result;
                        resultArray[batchoffset + i] *= coeff;
                    }

                    sumLoss += localloss / teachSignal[k].Length;
                }

                resultLoss += sumLoss / input[k].BatchCount;

                input[k].Grad = resultArray;
            }

            resultLoss /= input.Length;
            return resultLoss;
        }
    }
}