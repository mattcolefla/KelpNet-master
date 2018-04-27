#region License

/*

Copyright (c) 2009 - 2013 Fatjon Sakiqi

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

*/

#endregion

namespace Cloo
{
    using System;
    using System.Diagnostics;
    using System.Threading;
    using Cloo.Bindings;
    using ReflectSoftware.Insight;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Represents the parent type to any Cloo event types. </summary>
    ///
    /// <seealso cref="T:Cloo.ComputeResource"/>
    /// <seealso cref="ComputeEvent"/>
    /// <seealso cref="ComputeUserEvent"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public abstract class ComputeEventBase : ComputeResource
    {
        #region Fields

        /// <summary>   Occurs when aborted. </summary>
        private event ComputeCommandStatusChanged aborted;        
        /// <summary>   Occurs when completed. </summary>
        private event ComputeCommandStatusChanged completed;
        
        /// <summary>   The status. </summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private ComputeCommandStatusArgs status;
        
        /// <summary>   The status notify. </summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private ComputeEventCallback statusNotify;

        #endregion

        #region Events

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Occurs when the command associated with the event is abnormally terminated.
        /// </summary>
        ///
        /// ### <remarks>   Requires OpenCL 1.1. </remarks>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public event ComputeCommandStatusChanged Aborted
        {
            add
            {
                aborted += value;
                if (status != null && status.Status != ComputeCommandExecutionStatus.Complete)
                    value?.Invoke(this, status);
            }
            remove => aborted -= value;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Occurs when <c>ComputeEventBase.Status</c> changes to
        /// <c>ComputeCommandExecutionStatus.Complete</c>.
        /// </summary>
        ///
        /// ### <remarks>   Requires OpenCL 1.1. </remarks>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public event ComputeCommandStatusChanged Completed
        {
            add
            {
                completed += value;
                if (status != null && status.Status == ComputeCommandExecutionStatus.Complete)
                    value?.Invoke(this, status);
            }
            remove => completed -= value;
        }

        #endregion

        #region Properties

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   The handle of the <see cref="ComputeEventBase"/>. </summary>
        ///
        /// <value> The handle. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public CLEventHandle Handle
        {
            get;
            protected set;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeContext"/> associated with the <see cref="ComputeEventBase"/>.
        /// </summary>
        ///
        /// <value>
        /// The <see cref="ComputeContext"/> associated with the <see cref="ComputeEventBase"/>.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeContext Context { get; protected set; }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeDevice"/> time counter in nanoseconds when the associated command
        /// has finished execution.
        /// </summary>
        ///
        /// <value>
        /// The <see cref="ComputeDevice"/> time counter in nanoseconds when the associated command has
        /// finished execution.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long FinishTime => GetInfo<CLEventHandle, ComputeCommandProfilingInfo, long>(Handle, ComputeCommandProfilingInfo.Ended, CL12.GetEventProfilingInfo);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeDevice"/> time counter in nanoseconds when the associated command
        /// is enqueued in the <see cref="ComputeCommandQueue"/> by the host.
        /// </summary>
        ///
        /// <value>
        /// The <see cref="ComputeDevice"/> time counter in nanoseconds when the associated command is
        /// enqueued in the <see cref="ComputeCommandQueue"/> by the host.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long EnqueueTime => (long)GetInfo<CLEventHandle, ComputeCommandProfilingInfo, long>(Handle, ComputeCommandProfilingInfo.Queued, CL12.GetEventProfilingInfo);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the execution status of the associated command. </summary>
        ///
        /// <value>
        /// The execution status of the associated command or a negative value if the execution was
        /// abnormally terminated.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeCommandExecutionStatus Status => (ComputeCommandExecutionStatus)GetInfo<CLEventHandle, ComputeEventInfo, int>(Handle, ComputeEventInfo.ExecutionStatus, CL12.GetEventInfo);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeDevice"/> time counter in nanoseconds when the associated command
        /// starts execution.
        /// </summary>
        ///
        /// <value>
        /// The <see cref="ComputeDevice"/> time counter in nanoseconds when the associated command
        /// starts execution.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long StartTime => (long)GetInfo<CLEventHandle, ComputeCommandProfilingInfo, ulong>(Handle, ComputeCommandProfilingInfo.Started, CL12.GetEventProfilingInfo);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets the <see cref="ComputeDevice"/> time counter in nanoseconds when the associated command
        /// that has been enqueued is submitted by the host to the device.
        /// </summary>
        ///
        /// <value>
        /// The <see cref="ComputeDevice"/> time counter in nanoseconds when the associated command that
        /// has been enqueued is submitted by the host to the device.
        /// </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public long SubmitTime => (long)GetInfo<CLEventHandle, ComputeCommandProfilingInfo, ulong>(Handle, ComputeCommandProfilingInfo.Submitted, CL12.GetEventProfilingInfo);

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the <see cref="ComputeCommandType"/> associated with the event. </summary>
        ///
        /// <value> The <see cref="ComputeCommandType"/> associated with the event. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeCommandType Type { get; protected set; }

        #endregion
        
        #region Protected methods

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Releases the associated OpenCL object. </summary>
        ///
        /// <remarks>
        /// <paramref name="manual"/> must be <c>true</c> if this method is invoked directly by the
        /// application.
        /// </remarks>
        ///
        /// <param name="manual">   Specifies the operation mode of this method. </param>
        ///
        /// <seealso cref="M:Cloo.ComputeResource.Dispose(bool)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected override void Dispose(bool manual)
        {
            if (Handle.IsValid)
            {
                RILogManager.Default?.SendTrace("Dispose " + this + " in Thread(" + Thread.CurrentThread.ManagedThreadId + ").", "Information");
                CL12.ReleaseEvent(Handle);
                Handle.Invalidate();
            }
        }

        /// <summary>   Hook notifier. </summary>
        protected void HookNotifier()
        {
            statusNotify = StatusNotify;
            ComputeErrorCode error = CL11.SetEventCallback(Handle, (int)ComputeCommandExecutionStatus.Complete, statusNotify, IntPtr.Zero);
            ComputeException.ThrowOnError(error);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Executes the completed action. </summary>
        ///
        /// <param name="sender">   . </param>
        /// <param name="evArgs">   . </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected virtual void OnCompleted(object sender, ComputeCommandStatusArgs evArgs)
        {
            RILogManager.Default?.SendTrace(string.Intern("Complete ") + Type + string.Intern(" operation of ") + this + string.Intern("."), string.Intern("Information"));
            completed?.Invoke(sender, evArgs);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Executes the aborted action. </summary>
        ///
        /// <param name="sender">   . </param>
        /// <param name="evArgs">   . </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        protected virtual void OnAborted(object sender, ComputeCommandStatusArgs evArgs)
        {
            RILogManager.Default?.SendTrace(string.Intern("Abort ") + Type + string.Intern(" operation of ") + this + string.Intern("."), string.Intern("Information"));
            aborted?.Invoke(sender, evArgs);
        }

        #endregion

        #region Private methods

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Status notify. </summary>
        ///
        /// <param name="eventHandle">          Handle of the event. </param>
        /// <param name="cmdExecStatusOrErr">   The command execute status or error. </param>
        /// <param name="userData">             Information describing the user. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        private void StatusNotify(CLEventHandle eventHandle, int cmdExecStatusOrErr, IntPtr userData)
        {
            status = new ComputeCommandStatusArgs(this, (ComputeCommandExecutionStatus)cmdExecStatusOrErr);
            switch (cmdExecStatusOrErr)
            {
                case (int)ComputeCommandExecutionStatus.Complete: 
                    OnCompleted(this, status); 
                    break;
                default: OnAborted(this, status); 
                    break;
            }
        }

        #endregion
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Represents the arguments of a command status change. </summary>
    ///
    /// <seealso cref="T:System.EventArgs"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public sealed class ComputeCommandStatusArgs : EventArgs
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the event associated with the command that had its status changed. </summary>
        ///
        /// <value> The event. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeEventBase Event { get; private set; }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the execution status of the command represented by the event. </summary>
        ///
        /// <remarks>   Returns a negative integer if the command was abnormally terminated. </remarks>
        ///
        /// <value> The status. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeCommandExecutionStatus Status { get; private set; }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Creates a new <c>ComputeCommandStatusArgs</c> instance. </summary>
        ///
        /// <param name="ev">       The event representing the command that had its status changed. </param>
        /// <param name="status">   The status of the command. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeCommandStatusArgs(ComputeEventBase ev, ComputeCommandExecutionStatus status)
        {
            Event = ev;
            Status = status;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Creates a new <c>ComputeCommandStatusArgs</c> instance. </summary>
        ///
        /// <param name="ev">       The event of the command that had its status changed. </param>
        /// <param name="status">   The status of the command. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeCommandStatusArgs(ComputeEventBase ev, int status)
            : this(ev, (ComputeCommandExecutionStatus)status)
        { }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Calculates the command status changed. </summary>
    ///
    /// <param name="sender">   . </param>
    /// <param name="args">     . </param>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public delegate void ComputeCommandStatusChanged(object sender, ComputeCommandStatusArgs args);
}