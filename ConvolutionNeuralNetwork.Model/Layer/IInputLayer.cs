using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Model.Layer
{
    public interface IInputLayer<in TIn, out TOut> : ILayer<TIn, TOut>
        where TIn : IArray
        where TOut : IArray
    {
    }
}
