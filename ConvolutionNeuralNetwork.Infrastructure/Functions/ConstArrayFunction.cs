
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Infrastructure.Functions
{
    public class ConstArrayFunction : ConstActivationFunction, IArrayActivationFunction
    {
        public Array3D Activate(Array3D input)
        {
            return input;
        }

        public Array3D Derivative(Array3D input)
        {
            var result = new Array3D(input.Height, input.Width, input.Depth);
            for (var i = 0; i < result.Height; i++)
                for (var j = 0; j < result.Width; j++)
                    for (var k = 0; k < result.Depth; k++)
                        result[i, j, k] = 1;

            return result;
        }
    }
}
