
namespace ConvolutionNeuralNetwork.Infrastructure.Functions
{
    public class ConstActivationFunction : IActivationFunction
    {
        public double Activate(double value)
        {
            return value;
        }

        public double Derivative(double value)
        {
            return 1;
        }
    }
}
