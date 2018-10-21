using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Infrastructure.Functions
{
    public interface IArrayActivationFunction : IActivationFunction
    {
        Array3D Activate(Array3D input);
        Array3D Derivative(Array3D input);
    }
}
