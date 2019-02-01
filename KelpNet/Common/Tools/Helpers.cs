using System;

namespace KelpNet.Common.Tools
{
    public static class Helpers
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Format time span. </summary>
        ///
        /// <param name="span"> The span. </param>
        ///
        /// <returns>   The formatted time span. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static string FormatTimeSpan(TimeSpan span)
        {
            if (span.Days > 0)
            {
                if (span.Days == 1)
                    return
                        $"{span.Days} day, {span.Hours} hours, {span.Minutes} minutes, {span.Seconds} seconds, {span.Milliseconds} milliseconds";
                else
                    return
                        $"{span.Days} days, {span.Hours} hours, {span.Minutes} minutes, {span.Seconds} seconds, {span.Milliseconds} milliseconds";
            }
            
            if (span.Hours > 0)
            {
                if (span.Hours == 1)
                    return
                        $"{span.Hours} hour, {span.Minutes} minutes, {span.Seconds} seconds, {span.Milliseconds} milliseconds";
                else
                    return
                        $"{span.Hours} hours, {span.Minutes} minutes, {span.Seconds} seconds, {span.Milliseconds} milliseconds";
            }

            if (span.Minutes > 0)
            {
                if (span.Minutes == 1)
                    return
                        $"{span.Minutes} minute, {span.Seconds} seconds, {span.Milliseconds} milliseconds";
                else
                    return
                        $"{span.Minutes} minutes, {span.Seconds} seconds, {span.Milliseconds} milliseconds";
            }

            if (span.Seconds > 0)
            {
                if (span.Seconds == 1)
                    return
                        $"{span.Seconds} second, {span.Milliseconds} milliseconds";
                else
                    return
                        $"{span.Seconds} seconds, {span.Milliseconds} milliseconds";
            }

            return $"{span.TotalMilliseconds} ms";
        }
    }
}
