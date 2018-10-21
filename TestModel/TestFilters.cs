using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Infrastructure.Functions;
using ConvolutionNeuralNetwork.Model.Entity;
using ConvolutionNeuralNetwork.Model.Layer;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Tests.Helpers.Converters;

namespace TestModel
{
    [TestClass]
    public class TestFilters
    {
        private const string InitialDirectory = @"C:\Programming\ConvolutionNeuralNetwork\Images\Initial";
        private const string FilteredDirectory = @"C:\Programming\ConvolutionNeuralNetwork\Images\Filtered";

        [TestMethod]
        public void TestMethod1()
        {
            try
            {
                Directory.Delete(FilteredDirectory, true);
            }
            catch(Exception e) { Console.WriteLine("Path not deleted");}

            var filters = new List<double[,,]>
            {
                GetMask(2, 2, -1, 0, 0, 1), GetMask(2, 2, 0, -1, 1, 0),
                GetMask(2, 2, -1, 0, 0, -1), GetMask(2, 2, 0, -1, -1, 0),
                //GetMask(3, 3, 0, 0, 0 ,0, 1, 0, 0, 0, 0), GetMask(3, 3, 1, 1, 1, 1, 0, 1, 1, 1, 1)
            };

            foreach (var file in Directory.GetFiles(InitialDirectory))
            {
                var extStart = file.LastIndexOf('.');
                var nameStart = file.LastIndexOf('\\');
                var fileName = file.Substring(nameStart + 1, extStart - nameStart - 1);
                var fileExt = file.Substring(extStart + 1, file.Length  - extStart - 1);

                var destinationDirectory = Path.Combine(FilteredDirectory, fileName);
                Directory.CreateDirectory(destinationDirectory);

                var originCopyFullPath = Path.Combine(destinationDirectory, $"{fileName}.{fileExt}");

                File.Copy(file, originCopyFullPath);

                for (var i = 0; i < filters.Count; i += 2)
                {
                    ApplyFilter(originCopyFullPath, filters[i], filters[i + 1], Path.Combine(destinationDirectory, $"{fileName}_{i/2}.jpeg"));
                }
            }
        }

        /// <summary>
        ///
        /// </summary>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <param name="values"> Stored by width </param>
        /// <returns></returns>
        private double[,,] GetMask(int height, int width, params double[] values)
        {
            const int depth = 3;
            var array = new double[3,height, width];

            int index = 0;
            for (var i = 0; i < height; i++)
                for (var j = 0; j < width; j++, index++)
                    for (var k = 0; k < depth; k++)
                        array[k, i, j] = values[index];

            return array;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="originImage"></param>
        /// <param name="mask1"></param>
        /// <param name="mask2"></param>
        /// <param name="filteredImage">Full path to store applyed filter image</param>
        private void ApplyFilter(string originImage, double [,,] mask1, double [,,] mask2, string filteredImage)
        {
            Bitmap bitmap = new Bitmap(originImage);

            var image = new Array3D(bitmap.Height, bitmap.Width, 3);
            for (var i = 0; i < bitmap.Height; i++)
                for (var j = 0; j < bitmap.Width; j++)
                {
                    var pixel = bitmap.GetPixel(j, i);
                    image[i, j, 0] = pixel.R;
                    image[i, j, 1] = pixel.G;
                    image[i, j, 2] = pixel.B;
                }

            var filter = new ConvolutionalLayer
            {
                ActivationFunction = new ConstActivationFunction()
            };

            filter.Neurals.Add(new Filter { Weights = Converters.Convert(mask1) });
            filter.Neurals.Add(new Filter { Weights = Converters.Convert(mask2) });

            var output = filter.FormOutput(image) as Array3D;

            var g1 = output.ExtractLayer(0);
            var g2 = output.ExtractLayer(1);

            var resultArray = new double[g1.Height, g1.Width];
            for (var i = 0; i < g1.Height; i++)
                for (var j = 0; j < g1.Width; j++)
                    resultArray[i, j] = Math.Sqrt(Math.Pow(g1[i, j], 2) + Math.Pow(g2[i, j], 2));

            var imageBuffer = new byte[g1.Width * g1.Height * 4];
            var indexBytes = 0;

            for (var i = 0; i < g1.Height; i++)
                for (var j = 0; j < g1.Width; j++)
                {
                    imageBuffer[indexBytes++] = (byte)resultArray[i, j];
                    imageBuffer[indexBytes++] = 0;
                    imageBuffer[indexBytes++] = 0;
                    imageBuffer[indexBytes++] = 255;
                }

            unsafe
            {
                fixed (byte* ptr = imageBuffer)
                {
                    using (Bitmap outImage = new Bitmap(g1.Width, g1.Height, g1.Width * 4, PixelFormat.Format32bppPArgb,
                        new IntPtr(ptr)))
                    {
                        outImage.Save(filteredImage);
                    }
                }
            }
        }
    }
}
