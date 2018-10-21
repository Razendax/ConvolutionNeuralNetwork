using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Model.Helpers;

namespace ConvolutionNeuralNetwork.Model.Training
{
    public class CNNTrainer : NNTrainer<Array3D, Array3D>
    {
        public CNNTrainer(int epoch, int batch, IDataProvider<Array3D, Array3D> dataProvider) : base(epoch, batch, dataProvider)
        {
        }

        protected override Array3D ForwardPropagation(Array3D input, out List<Array3D> outputs)
        {
            throw new NotImplementedException();
        }

        protected override void BackPropagation(Array3D input, Array3D expected, Array3D actual, List<Array3D> outputs)
        {
            throw new NotImplementedException();
        }

        public override void Train(int epochs)
        {
            throw new NotImplementedException();
        }
    }
}
