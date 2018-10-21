using System;

namespace ConvolutionNeuralNetwork.Infrastructure.Functions
{
    public class SigmoidFunction : IActivationFunction
    {
        public SigmoidFunction(double alpha = 1)
        {
            _alpha = alpha;
        }

        private readonly double _alpha;
        
        public double Activate(double value)
        {
            return 1.0 / (1.0 + Math.Exp(- _alpha * value));
        }

        public double Derivative(double value)
        {
            return _alpha * value * (1.0 - value);
        }
    }
}
