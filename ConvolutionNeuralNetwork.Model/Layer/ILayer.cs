using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Model.Entity;

namespace ConvolutionNeuralNetwork.Model.Layer
{
    public interface ILayer<in TIn, out TOut>
        where TIn : IArray 
        where TOut : IArray
    {
    int NeuronCount { get; }
    TOut FormOutput(TIn input);
    List<INeuron> Neurals { get; }
    }
}
