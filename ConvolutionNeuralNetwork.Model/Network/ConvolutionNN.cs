using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Model.Network
{
    public class ConvolutionNN : INN<Array3D, Array3D>
    {
        public IList<string> Classes { get; set; }= new List<string>();

        public override Array3D ForwardPropagation(Array3D input)
        {
            var linput = InputLayer.FormOutput(input);
            foreach (var hidden in HiddenLayers)
            {
                linput = hidden.FormOutput(linput);
            }

            return OutputLayer.FormOutput(linput);
        }
    }
}
