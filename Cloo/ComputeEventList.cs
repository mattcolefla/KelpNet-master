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
    using System.Collections;
    using System.Collections.Generic;
    using System.Collections.ObjectModel;
    using System.Diagnostics;
    using Cloo.Bindings;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>   Represents a list of OpenCL generated or user created events. </summary>
    ///
    /// <seealso cref="T:System.Collections.Generic.IList{Cloo.ComputeEventBase}"/>
    /// <seealso cref="ComputeCommandQueue"/>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class ComputeEventList : IList<ComputeEventBase>
    {
        #region Fields

        /// <summary>   The events. </summary>
        [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
        private readonly List<ComputeEventBase> events;

        #endregion

        #region Constructors

        /// <summary>   Creates an empty <see cref="ComputeEventList"/>. </summary>
        public ComputeEventList()
        {
            events = new List<ComputeEventBase>();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Creates a new <see cref="ComputeEventList"/> from an existing list of
        /// <see cref="ComputeEventBase"/>s.
        /// </summary>
        ///
        /// <param name="events">   A list of <see cref="ComputeEventBase"/>s. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeEventList(IList<ComputeEventBase> events)
        {
            events = new Collection<ComputeEventBase>(events);
        }

        #endregion

        #region Properties

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the last <see cref="ComputeEventBase"/> on the list. </summary>
        ///
        /// <value> The last <see cref="ComputeEventBase"/> on the list. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeEventBase Last => events[events.Count - 1];

        #endregion

        #region Public methods

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Waits on the host thread for the specified events to complete. </summary>
        ///
        /// <param name="events">   The events to be waited for completition. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static void Wait(ICollection<ComputeEventBase> events)
        {
            int eventWaitListSize;
            CLEventHandle[] eventHandles = ComputeTools.ExtractHandles(events, out eventWaitListSize);
            ComputeErrorCode error = CL12.WaitForEvents(eventWaitListSize, eventHandles);
            ComputeException.ThrowOnError(error);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Waits on the host thread for the <see cref="ComputeEventBase"/>s in the
        /// <see cref="ComputeEventList"/> to complete.
        /// </summary>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void Wait()
        {
            ComputeEventList.Wait(events);
        }

        #endregion

        #region IList<ComputeEventBase> Members

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Index of the given item. </summary>
        ///
        /// <param name="item"> . </param>
        ///
        /// <returns>   An int. </returns>
        ///
        /// <seealso cref="M:System.Collections.Generic.IList{Cloo.ComputeEventBase}.IndexOf(ComputeEventBase)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public int IndexOf(ComputeEventBase item)
        {
            return events.IndexOf(item);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Inserts. </summary>
        ///
        /// <param name="index">    . </param>
        /// <param name="item">     . </param>
        ///
        /// <seealso cref="M:System.Collections.Generic.IList{Cloo.ComputeEventBase}.Insert(int,ComputeEventBase)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void Insert(int index, ComputeEventBase item)
        {
            events.Insert(index, item);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Removes at described by index. </summary>
        ///
        /// <param name="index">    . </param>
        ///
        /// <seealso cref="M:System.Collections.Generic.IList{Cloo.ComputeEventBase}.RemoveAt(int)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void RemoveAt(int index)
        {
            events.RemoveAt(index);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Indexer to get or set items within this collection using array index syntax.
        /// </summary>
        ///
        /// <param name="index">    . </param>
        ///
        /// <returns>   The indexed item. </returns>
        ///
        /// <seealso cref="M:System.Collections.Generic.IList{Cloo.ComputeEventBase}.this(int)"/>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeEventBase this[int index]
        {
            get => events[index];
            set => events[index] = value;
        }

        #endregion

        #region ICollection<ComputeEventBase> Members

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Adds item. </summary>
        ///
        /// <param name="item"> . </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void Add(ComputeEventBase item)
        {
            events.Add(item);
        }

        /// <summary>   Clears this object to its blank/initial state. </summary>
        public void Clear()
        {
            events.Clear();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Query if this object contains the given item. </summary>
        ///
        /// <param name="item"> . </param>
        ///
        /// <returns>   True if the object is in this collection, false if not. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public bool Contains(ComputeEventBase item)
        {
            return events.Contains(item);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Copies to. </summary>
        ///
        /// <param name="array">        . </param>
        /// <param name="arrayIndex">   . </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public void CopyTo(ComputeEventBase[] array, int arrayIndex)
        {
            events.CopyTo(array, arrayIndex);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the number of.  </summary>
        ///
        /// <value> The count. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public int Count => events.Count;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets a value indicating whether this object is read only. </summary>
        ///
        /// <value> True if this object is read only, false if not. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public bool IsReadOnly => false;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Removes the given item. </summary>
        ///
        /// <param name="item"> . </param>
        ///
        /// <returns>   True if it succeeds, false if it fails. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public bool Remove(ComputeEventBase item)
        {
            return events.Remove(item);
        }

        #endregion

        #region IEnumerable<ComputeEventBase> Members

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the enumerator. </summary>
        ///
        /// <returns>   The enumerator. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public IEnumerator<ComputeEventBase> GetEnumerator()
        {
            return ((IEnumerable<ComputeEventBase>)events).GetEnumerator();
        }

        #endregion

        #region IEnumerable Members

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the enumerator. </summary>
        ///
        /// <returns>   The enumerator. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable)events).GetEnumerator();
        }

        #endregion
    }
}