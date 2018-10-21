using System;
using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Model.Helpers
{
    public class WeightInitializer
    {
        private static readonly Random Rand = new Random();

        public static void Random(IEnumerable<IArray> neurals)
        {
            foreach (var neural in neurals)
            {
                var index = 0;
                var value = Rand.NextDouble();
                while (neural.SetValue(index, value))
                {
                }
            }
        }
    }
}
