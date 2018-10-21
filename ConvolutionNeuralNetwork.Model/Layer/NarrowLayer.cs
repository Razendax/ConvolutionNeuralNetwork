using System;
using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Model.Entity;
using Array = ConvolutionNeuralNetwork.Infrastructure.DataContainer.Array;

namespace ConvolutionNeuralNetwork.Model.Layer
{
    public class NarrowLayer : ITrainableLayer<Array3D, Array3D>
    {
        public int NeuronCount { get; } = 0;
        public List<INeuron> Neurals => null;
        public Array3D FormOutput(Array3D input)
        {
            var output = new Array3D(input.Height * input.Width * input.Depth);
            var index = 0;
            foreach (double value in input)
            {
                output[index++] = value;
            }

            return output;
        }

        public Array3D Train(Array3D error, Array3D input, Array3D output)
        {
            var index = 0;
            for (var i = 0; i < input.Height; i++)
                for (var j = 0; j < input.Width; j++)
                    for (var k = 0; k < input.Depth; k++)
                        input[i, j, k] = error[index++];

            return input;
        }

        public IArray Train(IArray error, IArray input, IArray output)
        {
            if (!(error is Array lError && input is Array3D lInput && output is Array lOutput))
                throw new InvalidCastException("Not appropriate cast");

            return Train(lError, lInput, lOutput);
        }

        public IArray FormOutput(IArray input)
        {
            if (!(input is Array3D lInput))
                throw new InvalidCastException("Not appropriate cast");

            return FormOutput(lInput);
        }
    }
}
