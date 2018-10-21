using System;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Infrastructure.Functions
{
    public class SoftMaxFunction : IArrayActivationFunction
    {
        private Array3D _input;
        private double _summ = 0.0;

        public Array3D Input
        {
            get => _input;
            set
            {
                _input = value;
                _summ = 0.0;
                foreach (double t in _input)
                    _summ += Math.Exp(t);
            }
        }

        public double Activate(double value)
        {
            return Math.Exp(value) / _summ;
        }

        public Array3D Activate(Array3D input)
        {
            Input = input;
            var result = new Array3D(Input.Height, Input.Width, Input.Depth);

            for (var i = 0; i < Input.Height; i++)
                for (var j = 0; j < Input.Width; j++)
                    for (var k = 0; k < Input.Depth; k++)
                        result[i, j, k] = Activate(Input[i, j, k]);

            return result;
        }

        public double Derivative(double value)
        {
            return value * (1 - value);
        }

        public Array3D Derivative(Array3D input)
        {
            Input = input;
            var result = new Array3D(Input.Height, Input.Width, Input.Depth);

            for (var i = 0; i < Input.Height; i++)
                for (var j = 0; j < Input.Width; j++)
                    for (var k = 0; k < Input.Depth; k++)
                        result[i, j, k] = Derivative(_input[i, j, k]);

            return result;
        }
    }
}
