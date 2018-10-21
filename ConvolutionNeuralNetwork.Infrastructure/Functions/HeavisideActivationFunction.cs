
namespace ConvolutionNeuralNetwork.Infrastructure.Functions
{
    public class HeavisideActivationFunction : IActivationFunction
    {
        private readonly double _min;
        private readonly double _max;
        private readonly double _threshold;

        public HeavisideActivationFunction(double min = 0, double max = 1, double threshold = 0.7)
        {
            _min = min;
            _max = max;
            _threshold = threshold;
        }

        public double Activate(double value)
        {
            return value < _threshold ? _min : _max;
        }

        public double Derivative(double value)
        {
            return 0;
        }
    }
}
