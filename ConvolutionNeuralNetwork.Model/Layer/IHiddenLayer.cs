using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Model.Entity;

namespace ConvolutionNeuralNetwork.Model.Layer
{
    public interface IHiddenLayer<TIn, TOut> : ITrainableLayer<TIn, TOut>
        where TIn : IArray
        where TOut : IArray
    {
    }
}
