using System;

namespace ConvolutionNeuralNetwork.Infrastructure.Functions
{
    public class ReLUFunction : IActivationFunction
    {
        public double Activate(double value)
        {
            return Math.Max(0.0, value);
        }

        public double Derivative(double value)
        {
            return value > 0 ? 1 : 0;
        }
    }
}
