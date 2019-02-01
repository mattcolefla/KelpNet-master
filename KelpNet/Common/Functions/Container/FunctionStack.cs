using System;
using System.Collections.Generic;
using System.Text;
using KelpNet.Common.Functions.Type;
using KelpNet.Common.Optimizers;
using ReflectSoftware.Insight;

namespace KelpNet.Common.Functions.Container
{
    using System.Linq;
    using JetBrains.Annotations;


    [Serializable]
    public class DeepNetwork : NetworkLayer
    {
        /// <summary>   Name of the function. </summary>
        const string FUNCTION_NAME = "DeepNetwork";

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   All layers are stored here as Function class. </summary>
        ///
        /// <value> The functions. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public NetworkLayer[] Layers { get; private set; }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Common.Functions.Container.DeepNetwork class.
        /// </summary>
        ///
        /// <param name="functions">    The functions. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public DeepNetwork([CanBeNull] NetworkLayer[] functions, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            Layers = functions;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Common.Functions.Container.DeepNetwork class.
        /// </summary>
        ///
        /// <param name="function">     The function. </param>
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public DeepNetwork([CanBeNull] NetworkLayer function, [CanBeNull] string name = FUNCTION_NAME, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null) : base(name, inputNames, outputNames)
        {
            Layers = new[] { function };
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Common.Functions.Container.DeepNetwork class.
        /// </summary>
        ///
        /// <param name="functions">    A variable-length parameters list containing functions. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public DeepNetwork([CanBeNull] params NetworkLayer[] functions) : base(FUNCTION_NAME)
        {
            Layers = new NetworkLayer[] { };
            Add(functions);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// It is not an efficient implementation because it is not assumed to be used frequently.
        /// </summary>
        ///
        /// <param name="function"> A variable-length parameters list containing function. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void Add([CanBeNull] params NetworkLayer[] function)
        {
            if (function != null && function.Length > 0)
            {
                List<NetworkLayer> functionList = new List<NetworkLayer>();

                if (Layers != null)
                {
                    functionList.AddRange(Layers);
                }

                functionList.AddRange(function.Where(t => t != null));

                Layers = functionList.ToArray();

                InputNames = Layers[0].InputNames;
                OutputNames = Layers[Layers.Length - 1].OutputNames;
            }
        }

        /// <summary>   Compress this object. </summary>
        public void Compress()
        {
            List<NetworkLayer> functionList = new List<NetworkLayer>(Layers);

            // compress layer
            for (int i = 0; i < functionList.Count - 1; i++)
            {
                if (functionList[i] is CompressibleLayer)
                {
                    if (functionList[i + 1] is CompressibleActivationLayer)
                    {
                        ((CompressibleLayer)functionList[i]).SetActivation((CompressibleActivationLayer)functionList[i + 1]);
                        functionList.RemoveAt(i + 1);
                    }
                }
            }

            Layers = functionList.ToArray();
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
            return Layers.Aggregate(xs, (current, t) => t.Forward(verbose, current));
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

        public override void Update(bool verbose = true)
        {
            foreach (var function in Layers)
            {
                if (verbose)
                    RILogManager.Default?.SendDebug("Updating " + function.Name);
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

        public override void ResetState(bool verbose = true)
        {
            foreach (NetworkLayer function in Layers)
            {
                if (verbose)
                    RILogManager.Default?.SendDebug("Resetting " + function.Name + " State");
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
        public override NdArray[] Predict(bool verbose = true, [CanBeNull] params NdArray[] xs)
        {
            return Layers.Aggregate(xs, (current, t) => t.Predict(verbose, current));
        }

        public string Describe()
        {
            string temp = string.Empty;
            foreach (NetworkLayer nl in Layers)
            {

            }

            return temp;
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
            foreach (NetworkLayer function in Layers)
            {
                RILogManager.Default?.SendDebug("Setting " + function.Name + " Optimizer");
                function.SetOptimizer(optimizers);
            }
        }
    }

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
        /// <summary>   Gets or sets the name. </summary>
        ///
        /// <value> The name. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public new string Name
        {
            set => base.Name = value;
            get => base.Name;
        }

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
        /// <param name="name">         (Optional) The name. </param>
        /// <param name="functions">    A variable-length parameters list containing functions. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public FunctionStack([CanBeNull] string name = FUNCTION_NAME, [CanBeNull] params Function[] functions) : base(name)
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
        public override NdArray[] Forward(bool verbose = true, [CanBeNull] params NdArray[] xs)
        {
            return Functions.Aggregate(xs, (current, t) => t.Forward(verbose, current));
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
            foreach (var function in Functions)
            {
                function.Update();
            }
        }

        public string Describe()
        {
            StringBuilder result = new StringBuilder();

            foreach (Function f in Functions)
            {
                result.AppendLine(" - Function: " + f.Name);

                if (f.InputNames != null)
                {
                    foreach (string s in f.InputNames)
                        result.AppendLine(" - Input: " + s);
                }

                if (f.OutputNames != null)
                {
                    foreach (string s in f.OutputNames)
                        result.AppendLine(" - Output: " + s);
                }

                foreach (Optimizer o in f.Optimizers)
                {
                    result.AppendLine(" - Optimizer: " + o.NAME);
                }
            }

            return result.ToString();
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
        public override NdArray[] Predict(bool verbose = true, [CanBeNull] params NdArray[] xs)
        {
            return Functions.Aggregate(xs, (current, t) => t.Predict(verbose, current));
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
