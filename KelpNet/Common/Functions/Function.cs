using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet.Common.Optimizers;
using KelpNet.Common.Tools;

namespace KelpNet.Common.Functions
{
    using JetBrains.Annotations;

    /// <summary>   Base class of Function stacked in FunctionStack. </summary>
    [Serializable]
    public abstract class Function
    {
        /// <summary>   The name. </summary>
        public string Name;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets a value indicating whether this object is GPU enable. </summary>
        ///
        /// <value> True if GPU enable, false if not. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public bool GpuEnable { get; protected set; }

        /// <summary>   Options for controlling the operation. </summary>
        public NdArray[] Parameters = { };
        /// <summary>   The optimizers. </summary>
        public Optimizer[] Optimizers = { };

        /// <summary>   The previous inputs. </summary>
        [NonSerialized]
        public List<NdArray[]> PrevInputs = new List<NdArray[]>();

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Forwards the given xs. </summary>
        ///
        /// <param name="xs">   A variable-length parameters list containing xs. </param>
        ///
        /// <returns>   A NdArray[]. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public abstract NdArray[] Forward(params NdArray[] xs);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backwards the given ys. </summary>
        ///
        /// <param name="ys">   A variable-length parameters list containing ys. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public virtual void Backward([CanBeNull] params NdArray[] ys){}

        /// <summary>   List of names of the inputs. </summary>
        public string[] InputNames;
        /// <summary>   List of names of the outputs. </summary>
        public string[] OutputNames;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Common.Functions.Function class.
        /// </summary>
        ///
        /// <param name="name">         The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected Function([CanBeNull] string name, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null)
        {
            Name = name;

            if (inputNames != null)
            {
                InputNames = inputNames.ToArray();
            }

            if (outputNames != null)
            {
                OutputNames = outputNames.ToArray();
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Sets an optimizer. </summary>
        ///
        /// <param name="optimizers">   A variable-length parameters list containing optimizers. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public virtual void SetOptimizer([NotNull] params Optimizer[] optimizers)
        {
            Optimizers = optimizers;

            foreach (Optimizer optimizer in optimizers)
            {
                optimizer?.AddFunctionParameters(Parameters);
            }
        }

        /// <summary>   Function to call when updating parameters) </summary>
        protected void BackwardCountUp()
        {
            foreach (NdArray parameter in Parameters)
            {
                parameter.CountUp();
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Evaluation function. </summary>
        ///
        /// <param name="input">    A variable-length parameters list containing input. </param>
        ///
        /// <returns>   A NdArray[]. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [CanBeNull]
        public virtual NdArray[] Predict([CanBeNull] params NdArray[] input)
        {
            return Forward(input);
        }

        /// <summary>   Updates this object. </summary>
        public virtual void Update()
        {
            foreach (Optimizer optimizer in Optimizers)
            {
                optimizer.Update();
            }
        }

        /// <summary>   Initialize input data that could not be used up by RNN etc. </summary>
        public virtual void ResetState()
        {
            PrevInputs = new List<NdArray[]>();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Returns a string that represents the current object. </summary>
        ///
        /// <returns>   A string that represents the current object. </returns>
        ///
        /// <seealso cref="M:System.Object.ToString()"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public override string ToString()
        {
            return Name;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Makes a deep copy of this object. </summary>
        ///
        /// <returns>   A copy of this object. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        [CanBeNull]
        public Function Clone()
        {
            return DeepCopyHelper.DeepCopy(this);
        }
    }
}
