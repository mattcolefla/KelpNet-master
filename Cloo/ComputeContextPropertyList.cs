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
    using System.Collections;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <inheritdoc />
    ///  <summary>   Represents a list of <see cref="T:Cloo.ComputeContextProperty" />s. </summary>
    ///  <remarks>
    ///  A <see cref="T:Cloo.ComputeContextPropertyList" /> is used to specify the properties of a
    ///  <see cref="T:Cloo.ComputeContext" />.
    ///  </remarks>
    ///  <seealso cref="T:System.Collections.Generic.ICollection{Cloo.ComputeContextProperty}" />
    ///  <seealso cref="T:Cloo.ComputeContext" />
    ///  <seealso cref="T:Cloo.ComputeContextProperty" />

    public sealed class ComputeContextPropertyList: ICollection<ComputeContextProperty>
    {
        #region Fields

        /// <summary>   The properties. </summary>
        [DebuggerBrowsable(DebuggerBrowsableState.Never)]
        private IList<ComputeContextProperty> properties;

        #endregion

        #region Constructors

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Creates a new <see cref="ComputeContextPropertyList"/> which contains a single item
        /// specifying a <see cref="ComputePlatform"/>.
        /// </summary>
        ///
        /// <param name="platform"> A <see cref="ComputePlatform"/>. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeContextPropertyList(ComputePlatform platform)
        {
            properties = new List<ComputeContextProperty>
            {
                new ComputeContextProperty(ComputeContextPropertyName.Platform, platform.Handle.Value)
            };
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Creates a new <see cref="ComputeContextPropertyList"/> which contains the specified
        /// <see cref="ComputeContextProperty"/>s.
        /// </summary>
        ///
        /// <param name="properties">   An enumerable of <see cref="ComputeContextProperty"/>'s. </param>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeContextPropertyList(IEnumerable<ComputeContextProperty> properties)
        {
            this.properties = new List<ComputeContextProperty>(properties);
        }

        #endregion

        #region Public methods

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Gets a <see cref="ComputeContextProperty"/> of a specified <c>ComputeContextPropertyName</c>.
        /// </summary>
        ///
        /// <param name="name"> The <see cref="ComputeContextPropertyName"/> of the
        ///                     <see cref="ComputeContextProperty"/>. </param>
        ///
        /// <returns>
        /// The requested <see cref="ComputeContextProperty"/> or <c>null</c> if no such
        /// <see cref="ComputeContextProperty"/> exists in the <see cref="ComputeContextPropertyList"/>.
        /// </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public ComputeContextProperty GetByName(ComputeContextPropertyName name)
        {
            return properties.FirstOrDefault(property => property.Name == name);
        }

        #endregion

        #region Internal methods

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Converts this object to an int pointer array. </summary>
        ///
        /// <returns>   This object as an IntPtr[]. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        internal IntPtr[] ToIntPtrArray()
        {
            IntPtr[] result = new IntPtr[2 * properties.Count + 1];
            for (int i = 0; i < properties.Count; i++)
            {
                result[2 * i] = new IntPtr((int)properties[i].Name);
                result[2 * i + 1] = properties[i].Value;
            }
            result[result.Length - 1] = IntPtr.Zero;
            return result;
        }

        #endregion

        #region ICollection<ComputeContextProperty> Members

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        ///  <summary>   Adds item. </summary>
        ///  <param name="item"> . </param>
        ///  <seealso cref="M:System.Collections.Generic.ICollection{Cloo.ComputeContextProperty}.Add(ComputeContextProperty)" />

        public void Add(ComputeContextProperty item)
        {
            properties.Add(item);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        ///  <summary>   Clears this object to its blank/initial state. </summary>
        ///  <seealso cref="M:System.Collections.Generic.ICollection{Cloo.ComputeContextProperty}.Clear()" />

        public void Clear()
        {
            properties.Clear();
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        ///  <summary>   Query if this object contains the given item. </summary>
        ///  <param name="item"> . </param>
        ///  <returns>   True if the object is in this collection, false if not. </returns>
        ///  <seealso cref="M:System.Collections.Generic.ICollection{Cloo.ComputeContextProperty}.Contains(ComputeContextProperty)" />

        public bool Contains(ComputeContextProperty item)
        {
            return properties.Contains(item);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        ///  <summary>   Copies to. </summary>
        ///  <param name="array">        . </param>
        ///  <param name="arrayIndex">   . </param>
        ///  <seealso cref="M:System.Collections.Generic.ICollection{Cloo.ComputeContextProperty}.CopyTo(ComputeContextProperty[],int)" />

        public void CopyTo(ComputeContextProperty[] array, int arrayIndex)
        {
            properties.CopyTo(array, arrayIndex);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        ///  <summary>   Gets the number of.  </summary>
        ///  <value> The count. </value>
        ///  <seealso cref="P:System.Collections.Generic.ICollection{Cloo.ComputeContextProperty}.Count" />

        public int Count => properties.Count;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        ///  <summary>   Gets a value indicating whether this object is read only. </summary>
        ///  <value> True if this object is read only, false if not. </value>
        ///  <seealso cref="P:System.Collections.Generic.ICollection{Cloo.ComputeContextProperty}.IsReadOnly" />

        public bool IsReadOnly => false;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <inheritdoc />
        ///  <summary>   Removes the given item. </summary>
        ///  <param name="item"> . </param>
        ///  <returns>   True if it succeeds, false if it fails. </returns>
        ///  <seealso cref="M:System.Collections.Generic.ICollection{Cloo.ComputeContextProperty}.Remove(ComputeContextProperty)" />

        public bool Remove(ComputeContextProperty item)
        {
            return properties.Remove(item);
        }

        #endregion

        #region IEnumerable<ComputeContextProperty> Members

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the enumerator. </summary>
        ///
        /// <returns>   The enumerator. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public IEnumerator<ComputeContextProperty> GetEnumerator()
        {
            return properties.GetEnumerator();
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
            return ((IEnumerable)properties).GetEnumerator();
        }

        #endregion
    }
}