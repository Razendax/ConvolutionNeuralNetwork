using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Model.Layer
{
    public interface IOutputLayer<in TIn, TOut> : ILayer<TIn, TOut>
        where TIn : IArray
        where TOut : IArray
    {
        double GetNetworkError(TOut expected, TOut actual);
        TOut CalculateError(TOut expected, TOut actual);
    }
}
