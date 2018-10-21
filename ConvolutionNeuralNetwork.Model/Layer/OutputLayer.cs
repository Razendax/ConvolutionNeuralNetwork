using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Infrastructure.Functions;
using ConvolutionNeuralNetwork.Model.Entity;

namespace ConvolutionNeuralNetwork.Model.Layer
{
    public class OutputLayer : IOutputLayer <Array3D, Array3D>
    {
        public IArrayActivationFunction ActivationFunction { get; set; } = new SoftMaxFunction();

        public IErrorFunction ErrorFunction { get; set; } = new CrossEntropyErrorFunction();

        public OutputLayer(int outputs)
        {
            NeuronCount = outputs;
        }

        public int NeuronCount { get; }

        public List<INeuron> Neurals => null;

        public Array3D FormOutput(Array3D input)
        {
            var output = ActivationFunction.Activate(input);

            return output;
        }

        public double GetNetworkError(Array3D expected, Array3D output)
        {
            return ErrorFunction.GetNetworkError(expected, output);
        }

        public Array3D CalculateError(Array3D expected, Array3D output)
        {
            var outputError = ErrorFunction.CalculateError(expected, output);
            return outputError * ActivationFunction.Derivative(output);
        }
    }
}
