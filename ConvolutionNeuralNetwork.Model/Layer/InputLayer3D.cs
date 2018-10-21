using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Model.Entity;

namespace ConvolutionNeuralNetwork.Model.Layer
{
    public class InputLayer3D : IInputLayer <Array3D, Array3D>
    {
        public int NeuronCount { get; }

        public InputLayer3D(int height, int width, int depth = 3)
        {
            NeuronCount = height * width * depth;
        }
        public List<INeuron> Neurals => null;

        public Array3D FormOutput(Array3D array)
        {
            return array;
        }
    }
}
