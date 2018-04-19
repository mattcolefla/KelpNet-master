﻿using System;
using System.Collections;
using System.IO;
using System.Linq;

namespace ChainerModelLoader
{
    /// <summary>   A npz format. </summary>
    class NpzFormat
    {
        public static void Load<T>(byte[] bytes, out T value) where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            using (var dict = Load<T>(bytes))
            {
                value = dict.Values.First();
            }
        }

        public static void Load<T>(string path, out T value) where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            using (var dict = Load<T>(path))
            {
                value = dict.Values.First();
            }
        }

        public static void Load<T>(Stream stream, out T value) where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            using (var dict = Load<T>(stream))
            {
                value = dict.Values.First();
            }
        }
        public static NpzDictionary<T> Load<T>(byte[] bytes) where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            return Load<T>(new MemoryStream(bytes));
        }

        public static NpzDictionary<T> Load<T>(string path, out NpzDictionary<T> value) where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            return value = Load<T>(new FileStream(path, FileMode.Open));
        }

        public static NpzDictionary<T> Load<T>(Stream stream, out NpzDictionary<T> value) where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            return value = Load<T>(stream);
        }

        public static NpzDictionary<T> Load<T>(string path) where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            return Load<T>(new FileStream(path, FileMode.Open));
        }

        public static NpzDictionary<T> Load<T>(Stream stream) where T : class, ICloneable, IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            return new NpzDictionary<T>(stream);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Loads a matrix. </summary>
        ///
        /// <param name="bytes">    The bytes. </param>
        ///
        /// <returns>   The matrix. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static NpzDictionary<Array> LoadMatrix(byte[] bytes)
        {
            return LoadMatrix(new MemoryStream(bytes));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Loads a matrix. </summary>
        ///
        /// <param name="path"> Full pathname of the file. </param>
        ///
        /// <returns>   The matrix. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static NpzDictionary<Array> LoadMatrix(string path)
        {
            return LoadMatrix(new FileStream(path, FileMode.Open));
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Loads a matrix. </summary>
        ///
        /// <param name="stream">   The stream. </param>
        ///
        /// <returns>   The matrix. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static NpzDictionary<Array> LoadMatrix(Stream stream)
        {
            return new NpzDictionary(stream);
        }
    }
}
