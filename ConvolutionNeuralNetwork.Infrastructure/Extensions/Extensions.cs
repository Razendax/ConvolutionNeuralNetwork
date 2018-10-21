using System;

namespace ConvolutionNeuralNetwork.Infrastructure.Extensions
{
    public static class Extensions
    {
        public static double NormalDistribution(this Random rand, double mean, double standardDeviation)
        {
            var r1 = 1 - rand.NextDouble();
            var r2 = 1 - rand.NextDouble();

            var y1 = Math.Sqrt(-2.0 * Math.Log(r1)) * Math.Cos(2.0 * Math.PI * r2);

            return mean + y1 * standardDeviation;
        }
    }
}
