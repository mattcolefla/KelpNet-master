using System;
using System.Collections.Generic;

namespace KelpNet.Common.Optimizers
{
    using JetBrains.Annotations;

    /// <summary>   Have parameters in the class that Optimizer is a source of. </summary>
    [Serializable]
    public abstract class Optimizer
    {
        private string Name = "Optimizer";
        public string NAME
        {
            get { return Name; }
            set { Name = value; }
        }

        /// <summary>   Number of updates. </summary>
        public long UpdateCount = 1;
        /// <summary>   Options for controlling the optimizer. </summary>
        protected List<OptimizerParameter> OptimizerParameters = new List<OptimizerParameter>();

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Adds a function parameters. </summary>
        ///
        /// <param name="functionParameters">   Options for controlling the function. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal abstract void AddFunctionParameters(NdArray[] functionParameters);

        /// <summary>   Updates this object. </summary>
        public void Update(bool verbose = true)
        {
            bool isUpdated = false;

            foreach (var t in OptimizerParameters)
            {
                // Run discount of slope and check if there was update
                if (t.FunctionParameter.Reduce(verbose))
                {
                    t.UpdateFunctionParameters(verbose);
                    t.FunctionParameter?.ClearGrad();
                    isUpdated = true;
                }
            }

            if (isUpdated)
            {
                UpdateCount++;
            }
        }

        /// <summary>   Resets the parameters. </summary>
        public void ResetParams()
        {
            foreach (var t in OptimizerParameters)
            {
                t.FunctionParameter?.ClearGrad();
            }

            UpdateCount = 0;
        }
    }

    /// <summary>   This class is created with FunctionParameter 1: 1. </summary>
    [Serializable]
    public abstract class OptimizerParameter
    {
        /// <summary>   The function parameter. </summary>
        public NdArray FunctionParameter;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Common.Optimizers.OptimizerParameter class.
        /// </summary>
        ///
        /// <param name="functionParameter">    The function parameter. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected OptimizerParameter([CanBeNull] NdArray functionParameter)
        {
            FunctionParameter = functionParameter;
        }

        /// <summary>   Updates the function parameters. </summary>
        public abstract void UpdateFunctionParameters(bool verbose);
    }
}
