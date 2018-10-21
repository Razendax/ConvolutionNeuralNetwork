using System.Drawing;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Model.Helpers
{
    public class OneIP : ImageProvider
    {
        public OneIP(string imageName, Array3D expected)
        {
            TrainPath = imageName;
            _expected = expected;
        }

        public override TrainingData<Array3D, Array3D> GetNext(bool isTraining)
        {
            if(_pair == null)
                Initialize();

            return _pair;
        }

        private void Initialize()
        {
            var image = Image.FromFile(TrainPath);
            Bitmap bitmap;
            if (ImageSize.Height == 0 || ImageSize.Width == 0)
                bitmap = new Bitmap(image, image.Width, image.Height);
            else
                bitmap = new Bitmap(image, ImageSize.Width, ImageSize.Height);

            var array = new Array3D(bitmap.Height, bitmap.Width, 3);
            for (var i = 0; i < bitmap.Height; i++)
            for (var j = 0; j < bitmap.Width; j++)
            {
                var pixel = bitmap.GetPixel(j, i);
                array[i, j, 0] = pixel.R / 255.0;
                array[i, j, 1] = pixel.G / 255.0;
                array[i, j, 2] = pixel.B / 255.0;
            }

            _pair = new TrainingData<Array3D, Array3D> { Expected = _expected, Input = array };
        }

        private readonly Array3D _expected;
        private TrainingData<Array3D, Array3D> _pair;
    }
}
