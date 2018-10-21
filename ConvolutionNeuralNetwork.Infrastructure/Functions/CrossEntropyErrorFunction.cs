using System;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Infrastructure.Functions
{
    public class CrossEntropyErrorFunction : IErrorFunction
    {
        public Array3D CalculateError(Array3D expected, Array3D actual)
        {
            return actual - expected;
        }

        public double GetNetworkError(Array3D expected, Array3D actual)
        {
            var result = 0.0;
            for (int i = 0; i < expected.Height; i++)
                for (var j = 0; j < expected.Width; j++)
                    for (var k = 0; k < expected.Depth; k++)
                        result += expected[i, j, k] * Math.Log(actual[i, j, k]);

            return -result;
        }
    }
}
