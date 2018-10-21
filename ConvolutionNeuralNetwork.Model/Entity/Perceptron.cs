using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Model.Entity
{
    public class Perceptron : INeuron
    {
        public Perceptron(int neurons)
        {
            Weights = new Array3D(neurons);
        }

        public Perceptron(Array3D weight)
        {
            Weights = weight.Clone() as Array3D;
        }

        public Perceptron(List<double> weight)
        {
            Weights = new Array3D(weight);
        }

        public override double FormOutput(Array3D input)
        {
            return (Weights * input).WeightSum() + Bias;
        }
    }
}
