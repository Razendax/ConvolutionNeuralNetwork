using System;

namespace ConvolutionNeuralNetwork.Infrastructure.Functions
{
    public class TanhActivationFunction : IActivationFunction
    {
        private readonly double _squiseX;
        private readonly double _squizeY;
        private readonly double _shiftX;
        private readonly double _shiftY;

        public TanhActivationFunction(double squizeX = 1, double squizeY = 1, double shiftX = 0, double shiftY = 0)
        {
            _squiseX = squizeX;
            _squizeY = squizeY;
            _shiftX = shiftX;
            _shiftY = shiftY;
        }

        public double Activate(double value)
        {
            return Math.Tanh(_squiseX * value + _shiftX) * _squizeY + _shiftY;
        }

        public double Derivative(double value)
        {
            return 1 - Math.Pow(value, 2);
        }
    }
}
