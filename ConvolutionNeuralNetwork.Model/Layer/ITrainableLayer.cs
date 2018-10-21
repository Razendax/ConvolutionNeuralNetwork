using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Model.Layer
{
    public interface ITrainableLayer
    {
        IArray Train(IArray error, IArray input, IArray output);
        IArray FormOutput(IArray input);
    }

    public interface ITrainableLayer<TIn, TOut> : ILayer<TIn, TOut>, ITrainableLayer
        where TIn : IArray
        where TOut : IArray
    {
        TIn Train(TOut error, TIn input, TOut output);
    }
}
