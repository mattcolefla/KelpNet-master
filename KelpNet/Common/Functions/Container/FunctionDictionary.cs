using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet.Common.Functions.Type;
using KelpNet.Common.Optimizers;

namespace KelpNet.Common.Functions.Container
{
    using JetBrains.Annotations;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <inheritdoc />
    ///  <summary>   (Serializable) dictionary of functions. </summary>
    ///  <seealso cref="T:KelpNet.Common.Functions.Function" />

    [Serializable]
    public sealed class FunctionDictionary : Function
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "FunctionDictionary";

        /// <summary>   Manage with FunctionRecord unit with I / O key added to function. </summary>
        public Dictionary<string, FunctionStack> FunctionBlockDictionary = new Dictionary<string, FunctionStack>();

        /// <summary>   a dictionary holding the name of the partitioning function. </summary>
        public Dictionary<string, FunctionStack> SplitedFunctionDictionary = new Dictionary<string, FunctionStack>();

        /// <summary>   dictionary execution order list. </summary>
        public List<FunctionStack> FunctionBlocks = new List<FunctionStack>();

        /// <summary>   True to compress. </summary>
        private readonly bool _compress = false;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Common.Functions.Container.FunctionDictionary
        /// class.
        /// </summary>
        ///
        /// <param name="compress">     (Optional) True to compress. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public FunctionDictionary(bool compress = false, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            _compress = compress;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Adds function. </summary>
        ///
        /// <param name="function"> The function to add. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void Add([NotNull] Function function)
        {
            if (_compress && // Do you want to summarize each branch
                (function is SingleInputFunction || function is MultiOutputFunction)) // Only one function with input is gathered
            {
                // Check if there is registration in dictionary with input name
                if (FunctionBlockDictionary.ContainsKey(function.InputNames[0]))
                {
                    // Concatenate to block if block already registered as dictionary
                    FunctionBlockDictionary[function.InputNames[0]].Add(function);

                    // Override output name
                    FunctionBlockDictionary[function.InputNames[0]].OutputNames = function.OutputNames.ToArray();

                    // Update split source output name if divided function
                    if (SplitedFunctionDictionary.ContainsKey(function.InputNames[0]))
                    {
                        FunctionStack spliteFunction = SplitedFunctionDictionary[function.InputNames[0]];

                        for (int i = 0; i < spliteFunction.OutputNames.Length; i++)
                        {
                            if (spliteFunction.OutputNames[i] == function.InputNames[0])
                            {
                                spliteFunction.OutputNames[i] = function.OutputNames[0];

                                if (!SplitedFunctionDictionary.ContainsKey(function.OutputNames[0]))
                                {
                                    SplitedFunctionDictionary.Add(function.OutputNames[0], spliteFunction);
                                }
                            }
                        }
                    }

                    if (!(function is MultiOutputFunction) && // If the output branches, do not register and cut the link
                      !FunctionBlockDictionary.ContainsKey(function.OutputNames[0])) // Do not register if already registered
                    {
                        // Add link to dictionary
                        FunctionBlockDictionary.Add(function.OutputNames[0], FunctionBlockDictionary[function.InputNames[0]]);
                    }
                    else if (function is SplitFunction splitFunction) 
                    {
                        var splitFunctions = splitFunction.SplitedFunctions;

                        for (int i = 0; i < splitFunctions.Length; i++)
                        {
                            // Add internal FunctionStack to link dictionary
                            FunctionBlockDictionary.Add(splitFunction.OutputNames[i], splitFunctions[i]);

                            // Add to SplitFunction's list
                            SplitedFunctionDictionary.Add(splitFunction.OutputNames[i], FunctionBlockDictionary[splitFunction.InputNames[0]]);
                        }
                    }

                    return;
                }
            }

            // less processing less compression, or processing for MultiInput, DualInput

            // whether the block is registered in the dictionary
            if (FunctionBlockDictionary.ContainsKey(function.OutputNames[0]))
            {
                // Concatenate to block if block already registered as dictionary
                FunctionBlockDictionary[function.OutputNames[0]].Add(function);
            }
            else
            {
                // Create a new block if not registered
                FunctionStack functionRecord = new FunctionStack(function, function.Name, function.InputNames, function.OutputNames);
                FunctionBlocks.Add(functionRecord);
                FunctionBlockDictionary.Add(function.Name, functionRecord);
            }
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
        public override NdArray[] Forward(bool verbose = true, [CanBeNull] params NdArray[] xs)
        {
            NdArray[] result = xs;

            Dictionary<string, NdArray> outPuts = new Dictionary<string, NdArray>();

            // Register the first data in the dictionary
            for (int i = 0; i < FunctionBlocks[0].InputNames.Length; i++)
            {
                outPuts.Add(FunctionBlocks[0].InputNames[i], xs[i]);
            }

            // Run in order of registration
            foreach (var t in FunctionBlocks)
            {
                string[] inputBlockNames = t.InputNames;

                // collect the input data

                // Implement the function
                result = t.Forward(verbose, inputBlockNames.Select(t1 => outPuts[t1]).ToArray());

                // Register the output data in the dictionary
                for (int j = 0; j < result.Length; j++)
                {
                    outPuts.Add(t.OutputNames[j], result[j]);
                }
            }

            return result;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backward. </summary>
        ///
        /// <param name="ys">   A variable-length parameters list containing ys. </param>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.Function.Backward(params NdArray[])"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override void Backward(bool verbose = true, [NotNull] params NdArray[] ys)
        {
            NdArray.Backward(verbose, ys[0]);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Weight update process. </summary>
        ///
        /// <seealso cref="M:KelpNet.Common.Functions.Function.Update()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override void Update()
        {
            foreach (var functionBlock in FunctionBlocks)
            {
                functionBlock.Update();
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
            foreach (var functionBlock in FunctionBlocks)
            {
                functionBlock.ResetState();
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
        public override NdArray[] Predict(bool verbose = true, [CanBeNull] params NdArray[] xs)
        {
            NdArray[] result = xs;

            // dictionary of output data
            Dictionary<string, NdArray> outPuts = new Dictionary<string, NdArray>();

            // Register the output data in the dictionary
            for (int j = 0; j < FunctionBlocks[0].InputNames.Length; j++)
            {
                outPuts.Add(FunctionBlocks[0].InputNames[j], xs[j]);
            }

            foreach (var t in FunctionBlocks)
            {
                string[] inputBlockNames = t.InputNames;

                // collect the input data

                // Implement the function
                result = t.Predict(verbose, inputBlockNames.Select(t1 => outPuts[t1]).ToArray());

                // Register the output data in the dictionary
                for (int j = 0; j < result.Length; j++)
                {
                    outPuts.Add(t.OutputNames[j], result[j]);
                }
            }

            return result;
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
            foreach (var functionBlock in FunctionBlocks)
            {
                functionBlock.SetOptimizer(optimizers);
            }
        }
    }
}
