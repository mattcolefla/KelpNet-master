using System;

namespace KelpNet.Common
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// Random number element In C #, when instantiating multiple Random simultaneously, only similar
    /// values are emitted It is necessary to manage it collectively in one place.
    /// </summary>
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    public class Mother
    {
#if DEBUG
        /// <summary>   Fix seed when debugging. </summary>
        public static Random Dice = new Random(128);
#else
        public static Random Dice = new Random();
#endif

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>   Gets the box muller 2. </summary>
        ///
        /// <value> The box muller 2. </value>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        static double Alpha, Beta, BoxMuller1, BoxMuller2;
        /// <summary>   True to flip. </summary>
        static bool Flip;
        /// <summary>   The mu. </summary>
        public static double Mu = 0;
        /// <summary>   The sigma. </summary>
        public static double Sigma = 1;

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Obtain a normal distribution random number with mean mu and standard deviation sigma. By the
        /// Box-Muller method。.
        /// </summary>
        ///
        /// <returns>   A double. </returns>
        ////////////////////////////////////////////////////////////////////////////////////////////////////

        public static double RandomNormal()
        {
            if (!Flip)
            {
                Alpha = Dice.NextDouble();
                Beta = Dice.NextDouble() * Math.PI * 2;
                BoxMuller1 = Math.Sqrt(-2 * Math.Log(Alpha));
                BoxMuller2 = Math.Sin(Beta);
            }
            else
            {
                BoxMuller2 = Math.Cos(Beta);
            }

            Flip = !Flip;

            return Sigma * (BoxMuller1 * BoxMuller2) + Mu;
        }
    }
}
