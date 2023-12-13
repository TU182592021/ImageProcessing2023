#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    printf("Input image is required.\n");
    return EXIT_FAILURE;
  }

  // 画像の読み込み
  cv::Mat img = cv::imread(argv[1], cv::IMREAD_ANYCOLOR);
  if (img.empty())
  { // 画像ファイルが見つからない場合の処理
    printf("Input image is not found.\n");
    return EXIT_FAILURE;
  }

  auto clip = [](int v)
  {
    if (v > 255)
      v = 255;
    if (v < 0)
      v = 0;
    return v;
  };

  auto GammaCorrection = [](int v, double gamma)
  {
    double in = static_cast<double>(v) / 255.0;
    double out = pow(in, gamma) * 255.0;
    return static_cast<int>(out);
  };
  auto Quantization = [](double v, double delta)
  {
    double x = static_cast<double>(v);
    double Qx = floor(x / delta + 0.5);
    return Qx;
  };

  auto Dequantization = [](int v, double delta)
  {
    double rec = v * delta;
    return static_cast<int>(rec);
  };

  const int W = img.cols;
  const int H = img.rows;
  uint8_t *pixels = img.data;
  const double delta = 1.0;

  for (int y = 0; y < H; y++)
  {
    for (int x = 0; x < W; x++)
    {
      int idx = y * W + x;
      int val = pixels[idx];
      val = Dequantization(Quantization(val, delta), delta);
      pixels[idx] = clip(val);
    }
  }

  // 画像の表示
  cv::imshow("window", img);
  // キー入力を待つ
  cv::waitKey();
  // 全てのウインドウを破棄
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}
