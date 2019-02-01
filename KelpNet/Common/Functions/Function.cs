using System;
using System.Collections.Generic;
using System.Linq;
using KelpNet.Common.Optimizers;
using KelpNet.Common.Tools;

namespace KelpNet.Common.Functions
{
    using JetBrains.Annotations;

    /// <inheritdoc />
    /// <summary>   Base class of NetworkLayer stacked in DeepLearningNetwork. </summary>
    [Serializable]
    public abstract class NetworkLayer : IComparable
    {
        /// <summary>   The name. </summary>
        public string Name;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets/Sets a value indicating whether this object is GPU enabled. </summary>
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
        /// <summary>   Forwards using the variable-length NdArray parameter. </summary>
        ///
        /// <param name="xs">   A variable-length parameter of type NdArray. </param>
        ///
        /// <returns>   A NdArray[]. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public abstract NdArray[] Forward(bool verbose, params NdArray[] xs);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backwards using the variable-length NdArray parameter. </summary>
        ///
        /// <param name="ys">   A variable-length parameter of type NdArray. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public virtual void Backward(bool verbose, [CanBeNull] params NdArray[] ys) { }

        /// <summary>   List of names of the inputs. </summary>
        public string[] InputNames;
        /// <summary>   List of names of the outputs. </summary>
        public string[] OutputNames;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Initializes a new instance of the KelpNet.Common.Functions.NetworkLayer class.
        /// </summary>
        ///
        /// <param name="name">         The name. </param>
        /// <param name="inputNames">   (Optional) List of names of the inputs. </param>
        /// <param name="outputNames">  (Optional) List of names of the outputs. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected NetworkLayer([CanBeNull] string name, [CanBeNull] string[] inputNames = null, [CanBeNull] string[] outputNames = null)
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
        public virtual NdArray[] Predict(bool verbose = true, [CanBeNull] params NdArray[] input)
        {
            return Forward(verbose, input);
        }

        /// <summary>   Updates this object. </summary>
        public virtual void Update(bool verbose = true)
        {
            foreach (Optimizer optimizer in Optimizers)
            {
                optimizer.Update();
            }
        }

        /// <summary>   Initialize input data that could not be used up by RNN etc. </summary>
        public virtual void ResetState(bool verbose = true)
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
        public NetworkLayer Clone()
        {
            return DeepCopyHelper.DeepCopy(this);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        ///  <summary>
        ///  Compares the current instance with another object of the same type and returns an integer
        ///  that indicates whether the current instance precedes, follows, or occurs in the same position
        ///  in the sort order as the other object.
        ///  </summary>
        ///  <exception cref="T:System.ArgumentException">   . </exception>
        ///  <param name="obj">  An object to compare with this instance. </param>
        ///  <returns>
        ///  A value that indicates the relative order of the objects being compared. The return value has
        ///  these meanings: Value Meaning Less than zero This instance precedes <paramref name="obj" />
        ///  in the sort order. Zero This instance occurs in the same position in the sort order as
        ///  <paramref name="obj" />. Greater than zero This instance follows <paramref name="obj" /> in
        ///  the sort order.
        ///  </returns>
        ///  <seealso cref="M:System.IComparable.CompareTo(object)" />

        public int CompareTo([NotNull] object obj)
        {
            Function f = (Function)obj;
            if (f.Name == Name)
                return 0;

            if (InputNames != null && OutputNames != null && f.InputNames != null && f.OutputNames != null)
            {
                if (InputNames.Length == f.InputNames.Length && OutputNames.Length == f.OutputNames.Length)
                    return (0);
            }

            return (-1);
        }
    }
    /// <inheritdoc />
    /// <summary>   Base class of Function stacked in FunctionStack. </summary>
    [Serializable]
    public abstract class Function : IComparable
    {
        /// <summary>   The name. </summary>
        public string Name;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets/Sets a value indicating whether this object is GPU enabled. </summary>
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
        /// <summary>   Forwards using the variable-length NdArray parameter. </summary>
        ///
        /// <param name="xs">   A variable-length parameter of type NdArray. </param>
        ///
        /// <returns>   A NdArray[]. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public abstract NdArray[] Forward(bool verbose, params NdArray[] xs);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Backwards using the variable-length NdArray parameter. </summary>
        ///
        /// <param name="ys">   A variable-length parameter of type NdArray. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public virtual void Backward(bool verbose, [CanBeNull] params NdArray[] ys){}

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
        public virtual NdArray[] Predict(bool verbose = true, [CanBeNull] params NdArray[] input)
        {
            return Forward(verbose, input);
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

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        ///  <summary>
        ///  Compares the current instance with another object of the same type and returns an integer
        ///  that indicates whether the current instance precedes, follows, or occurs in the same position
        ///  in the sort order as the other object.
        ///  </summary>
        ///  <exception cref="T:System.ArgumentException">   . </exception>
        ///  <param name="obj">  An object to compare with this instance. </param>
        ///  <returns>
        ///  A value that indicates the relative order of the objects being compared. The return value has
        ///  these meanings: Value Meaning Less than zero This instance precedes <paramref name="obj" />
        ///  in the sort order. Zero This instance occurs in the same position in the sort order as
        ///  <paramref name="obj" />. Greater than zero This instance follows <paramref name="obj" /> in
        ///  the sort order.
        ///  </returns>
        ///  <seealso cref="M:System.IComparable.CompareTo(object)" />

        public int CompareTo([NotNull] object obj)
        {
            Function f = (Function) obj;
            if (f.Name == Name)
                return 0;

            if (InputNames != null && OutputNames != null && f.InputNames != null && f.OutputNames != null)
            {
                if (InputNames.Length == f.InputNames.Length && OutputNames.Length == f.OutputNames.Length)
                    return (0);
            }

            return (-1);
        }
    }
}
