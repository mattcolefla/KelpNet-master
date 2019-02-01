using System;
using System.Collections.Generic;
using KelpNet.Common;
using KelpNet.Common.Functions;

namespace KelpNet.Functions.Noise
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   (Serializable) a stochastic depth. </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Function"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class StochasticDepth : Function // Use to replace SplitFunction
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "StochasticDepth";
        /// <summary>   The pl. </summary>
        private readonly Real _pl;
        /// <summary>   List of skips. </summary>
        private readonly List<bool> _skipList = new List<bool>();
        /// <summary>   Skipped by probability. </summary>
        private readonly Function _function;
        /// <summary>   Always executed. </summary>
        private readonly Function _resBlock;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Query if this object is skip. </summary>
        ///
        /// <returns>   True if skip, false if not. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private bool IsSkip()
        {
            bool result = Mother.Dice.NextDouble() >= _pl;
            _skipList.Add(result);
            return result;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Functions.Noise.StochasticDepth class.
        /// </summary>
        ///
        /// <param name="function">     The function. </param>
        /// <param name="resBlock">     (Optional) The resource block. </param>
        /// <param name="pl">           (Optional) The pl. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public StochasticDepth([CanBeNull] Function function, [CanBeNull] Function resBlock = null, double pl = 0.5, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            _function = function;
            _resBlock = resBlock;
            _pl = pl;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Forwards the given xs. </summary>
        ///
        /// <param name="verbose">  (Optional) True to verbose. </param>
        /// <param name="xs">       A variable-length parameters list containing xs. </param>
        ///
        /// <returns>   A NdArray[]. </returns>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.Function.Forward(params NdArray[])"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [NotNull]
        public override NdArray[] Forward(bool verbose = true, [CanBeNull] params NdArray[] xs)
        {
            List<NdArray> resultArray = new List<NdArray>();
            NdArray[] resResult = xs;

            if (_resBlock != null)
                resResult = _resBlock.Forward(verbose, xs);

            resultArray.AddRange(resResult);

            if (!IsSkip())
            {
                Real scale = 1 / (1 - _pl);
                NdArray[] result = _function.Forward(verbose, xs);

                foreach (var t in result)
                {
                    for (int j = 0; j < t.Data.Length; j++)
                        t.Data[j] *= scale;
                }

                resultArray.AddRange(result);
            }
            else
            {
                NdArray[] result = new NdArray[resResult.Length];

                for (int i = 0; i < result.Length; i++)
                    result[i] = new NdArray(resResult[i].Shape, resResult[i].BatchCount, resResult[i].ParentFunc);

                resultArray.AddRange(result);
            }

            return resultArray.ToArray();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backwards the given ys. </summary>
        ///
        /// <param name="ys">   A variable-length parameters list containing ys. </param>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.Function.Backward(params NdArray[])"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override void Backward(bool verbose = true, params NdArray[] ys)
        {
            _resBlock?.Backward(verbose, ys);
            bool isSkip = _skipList[_skipList.Count - 1];
            _skipList.RemoveAt(_skipList.Count - 1);

            if (!isSkip)
            {
                NdArray[] copyys = new NdArray[ys.Length];
                Real scale = 1 / (1 - _pl);

                for (int i = 0; i < ys.Length; i++)
                {
                    copyys[i] = ys[i].Clone();

                    for (int j = 0; j < ys[i].Data.Length; j++)
                        copyys[i].Data[j] *= scale;
                }

                _function.Backward(verbose, copyys);
            }
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

        public override NdArray[] Predict(bool verbose = true, params NdArray[] xs)
        {
            return _function.Predict(verbose, xs);
        }
    }
}
