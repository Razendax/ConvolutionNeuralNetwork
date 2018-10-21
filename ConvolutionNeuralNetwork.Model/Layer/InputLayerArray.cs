using System.Collections.Generic;
using ConvolutionNeuralNetwork.Model.Entity;
using Array = ConvolutionNeuralNetwork.Infrastructure.DataContainer.Array;

namespace ConvolutionNeuralNetwork.Model.Layer
{
    public class InputLayerArray : IInputLayer<Array, Array>
    {
        public InputLayerArray(int neurals)
        {
            NeuronCount = neurals;
        }

        public int NeuronCount { get; }
        public Array FormOutput(Array array)
        {
            return array;
        }

        public List<INeuron> Neurals => null;
    }
}
