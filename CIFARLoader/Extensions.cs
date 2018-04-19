namespace CIFARLoader
{
    /// <summary>   An extensions. </summary>
    internal static class Extensions
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   A string extension method that trim null. </summary>
        ///
        /// <param name="t">    The t to act on. </param>
        ///
        /// <returns>   A string. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static string TrimNull(this string t)
        {
            return t.Trim((char)0x20, (char)0x00);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   A string extension method that trim slash. </summary>
        ///
        /// <param name="t">    The t to act on. </param>
        ///
        /// <returns>   A string. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static string TrimSlash(this string t)
        {
            return t.TrimEnd('/');
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   A string extension method that trim volume. </summary>
        ///
        /// <param name="t">    The t to act on. </param>
        ///
        /// <returns>   A string. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static string TrimVolume(this string t)
        {
            if (t.Length > 3 && t[1] == ':' && t[2] == '/')
            {
                return t.Substring(3);
            }

            if (t.Length > 2 && t[0] == '/' && t[1] == '/')
            {
                return t.Substring(2);
            }

            return t;
        }
    }
}
