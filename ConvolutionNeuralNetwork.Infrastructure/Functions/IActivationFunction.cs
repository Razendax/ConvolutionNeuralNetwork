

namespace ConvolutionNeuralNetwork.Infrastructure.Functions
{
    public interface IActivationFunction
    {
        double Activate(double value);
        double Derivative(double value);
    }
}
