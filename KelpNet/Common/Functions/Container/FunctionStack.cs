using System;
using System.Collections.Generic;
using KelpNet.Common.Optimizers;

namespace KelpNet.Common.Functions.Container
{
    using System.Linq;
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// The main class of this library stacking layers a set of functions that are executed
    /// simultaneously in one Forward, Backward, Update.
    /// </summary>
    ///
    /// <seealso cref="T:KelpNet.Common.Functions.Function"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    [Serializable]
    public class FunctionStack : Function
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "FunctionStack";

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   All layers are stored here as Function class. </summary>
        ///
        /// <value> The functions. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public Function[] Functions { get; private set; }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Common.Functions.Container.FunctionStack class.
        /// </summary>
        ///
        /// <param name="functions">    The functions. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public FunctionStack([CanBeNull] Function[] functions, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            Functions = functions;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Common.Functions.Container.FunctionStack class.
        /// </summary>
        ///
        /// <param name="function">     The function. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public FunctionStack([CanBeNull] Function function, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            Functions = new[] { function };
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Common.Functions.Container.FunctionStack class.
        /// </summary>
        ///
        /// <param name="functions">    A variable-length parameters list containing functions. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public FunctionStack([CanBeNull] params Function[] functions) : base(FUNCTION_NAME)
        {
            Functions = new Function[]{};
            Add(functions);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// It is not an efficient implementation because it is not assumed to be used frequently.
        /// </summary>
        ///
        /// <param name="function"> A variable-length parameters list containing function. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void Add([CanBeNull] params Function[] function)
        {
            if (function != null && function.Length > 0)
            {
                List<Function> functionList = new List<Function>();

                if (Functions != null)
                {
                    functionList.AddRange(Functions);
                }

                functionList.AddRange(function.Where(t => t != null));

                Functions = functionList.ToArray();

                InputNames = Functions[0].InputNames;
                OutputNames = Functions[Functions.Length - 1].OutputNames;
            }
        }

        /// <summary>   Compress this object. </summary>
        public void Compress()
        {
            List<Function> functionList = new List<Function>(Functions);

            // compress layer
            for (int i = 0; i < functionList.Count - 1; i++)
            {
                if (functionList[i] is CompressibleFunction)
                {
                    if (functionList[i + 1] is CompressibleActivation)
                    {
                        ((CompressibleFunction)functionList[i]).SetActivation((CompressibleActivation)functionList[i + 1]);
                        functionList.RemoveAt(i + 1);
                    }
                }
            }

            Functions = functionList.ToArray();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Forward. </summary>
        ///
        /// <param name="xs">   A variable-length parameters list containing xs. </param>
        ///
        /// <returns>   A NdArray[]. </returns>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.Function.Forward(params NdArray[])"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [CanBeNull]
        public override NdArray[] Forward([CanBeNull] params NdArray[] xs)
        {
            return Functions.Aggregate(xs, (current, t) => t.Forward(current));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward. </summary>
        ///
        /// <param name="ys">   A variable-length parameters list containing ys. </param>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.Function.Backward(params NdArray[])"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override void Backward([NotNull] params NdArray[] ys)
        {
            NdArray.Backward(ys[0]);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Weight update process. </summary>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.Function.Update()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override void Update()
        {
            foreach (var function in Functions)
            {
                function.Update();
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Process to restore specific data to initial value after executing certain processing.
        /// </summary>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.Function.ResetState()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override void ResetState()
        {
            foreach (Function function in Functions)
            {
                function.ResetState();
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Run the forecast. </summary>
        ///
        /// <param name="xs">   A variable-length parameters list containing xs. </param>
        ///
        /// <returns>   A NdArray[]. </returns>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.Function.Predict(params NdArray[])"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [CanBeNull]
        public override NdArray[] Predict([CanBeNull] params NdArray[] xs)
        {
            return Functions.Aggregate(xs, (current, t) => t.Predict(current));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets an optimizer. </summary>
        ///
        /// <param name="optimizers">   A variable-length parameters list containing optimizers. </param>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.Function.SetOptimizer(params Optimizer[])"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override void SetOptimizer([CanBeNull] params Optimizer[] optimizers)
        {
            foreach (Function function in Functions)
            {
                function.SetOptimizer(optimizers);
            }
        }
    }
}
