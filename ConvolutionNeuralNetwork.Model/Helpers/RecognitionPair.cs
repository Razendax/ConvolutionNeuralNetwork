using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Model.Helpers
{
    public class RecognitionPair
    {
        public Array3D Image { get; set; }
        public string Class { get; set; }
    }
}
