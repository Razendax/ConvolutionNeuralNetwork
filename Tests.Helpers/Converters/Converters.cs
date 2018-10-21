

using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace Tests.Helpers.Converters
{
    public class Converters
    {
        public static Array3D Convert(double[,,] array)
        {
            var result = new double[array.GetLength(1), array.GetLength(2), array.GetLength(0)];
            for (var i = 0; i < array.GetLength(0); i++)
            for (var j = 0; j < array.GetLength(1); j++)
            for (var k = 0; k < array.GetLength(2); k++)
                result[j, k, i] = array[i, j, k];

            return new Array3D(result);
        }
    }
}
