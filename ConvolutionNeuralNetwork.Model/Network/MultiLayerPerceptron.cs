using System.Text;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Model.Network
{
    public class MultiLayerPerceptron : INN<Array3D, Array3D>
    {
        private readonly StringBuilder _message = new StringBuilder();
        public string Name { get; set; } = "Ole ole";

        public override Array3D ForwardPropagation(Array3D input)
        {
            IArray linput = InputLayer.FormOutput(input);
            foreach (var hidden in HiddenLayers)
            {
                linput = hidden.FormOutput(linput);
            }

            return OutputLayer.FormOutput(linput as Array3D);
        }
    }
}
