
namespace ConvolutionNeuralNetwork.Model.Helpers
{
    public class TrainingData<TIn, TExpected>
    {
        public TIn Input { get; set; }
        public TExpected Expected { get; set; }
    }
}
