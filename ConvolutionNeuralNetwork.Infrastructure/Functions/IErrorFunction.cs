using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Infrastructure.Functions
{
    public interface IErrorFunction
    {
        double GetNetworkError(Array3D expected, Array3D actual);
        Array3D CalculateError(Array3D expected, Array3D actual);
    }
}
