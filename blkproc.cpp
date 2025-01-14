#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <vector>

#include "qtables.hpp"
#include "tools.hpp"
#include "jpgheaders.hpp"
#include "time.hpp"

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    printf("Input image is required.\n");
    return EXIT_FAILURE;
  }
  if (argc < 3)
  {
    printf("Qfactor is missing.\n");
    return EXIT_FAILURE;
  }

  // 画像の読み込み
  cv::Mat rgb = cv::imread(argv[1], cv::IMREAD_ANYCOLOR);
  if (rgb.empty())
  { // 画像ファイルが見つからない場合の処理
    printf("Input image is not found.\n");
    return EXIT_FAILURE;
  }

  int QF = strtol(argv[2], nullptr, 10);
  if (QF < 0 || QF > 100)
  {
    printf("The valid range of qfactor is from 0 to 100.\n");
    return EXIT_FAILURE;
  }

  int qtables[2][DCTSIZE * DCTSIZE];
  create_qtable(0, QF, &qtables[0][0]);
  create_qtable(1, QF, &qtables[1][0]);

  int YCCtype = YUV411;
  bitstream enc;
  create_mainheader(rgb.cols, rgb.rows, rgb.channels(), qtables[0], qtables[1], YCCtype, enc);

  cv::Mat YCbCr = rgb.clone();
  cv::Mat recimg = rgb.clone();

  myBGR2YCbCr(rgb, YCbCr);
  std::vector<cv::Mat> buf(3); // Y: buf[0], Cb: buf[1], Cr: Buf[2]
  cv::split(YCbCr, buf);       // planar形式に変換
  std::vector<cv::Mat> buf_f(3);

  // ------ ENCODE
  double dct2_time = 0.0;
  double quantize_time = 0.0;
  for (size_t c = 0; c < buf.size(); ++c)
  {
    int Hl = (YCC_HV[YCCtype][0] & 0xF0) >> 4;
    int Vl = YCC_HV[YCCtype][0] & 0x0F;
    if (c > 0)
    {
      cv::resize(buf[c], buf[c], cv::Size(), 1.0 / Hl, 1.0 / Vl, cv::INTER_AREA);
    }
    buf[c].convertTo(buf_f[c], CV_32F); // 浮動小数点に変換
    buf_f[c] -= 128.0;
    auto s0 = time_start(); // 全画素の値から128を引く(DCレベルシフト)
    blkproc(buf_f[c], blk::dct2);
    auto d0 = time_stop(s0);
    auto s1 = time_start();                           // 順方向のDCT➀
    blkproc(buf_f[c], blk::quantize, qtables[c > 0]); // 量子化⓶
    auto d1 = time_stop(s1);
    dct2_time += static_cast<double>(d0) / 1000.0;
    quantize_time += static_cast<double>(d1) / 1000.0;
  }
  printf("dct2_time takes %f[ms]\nquantize_time takes %f[ms]\n", dct2_time, quantize_time);
  // ハフマン符号化⓷
  auto s2 = time_start();
  entropy_coding(buf_f, YCCtype, enc);
  auto d2 = time_stop(s2);
  double entropy_coding_time = static_cast<double>(d2) / 1000.0;
  printf("entropy_coding_time takes %f[ms]\n", entropy_coding_time);
  auto codestream = enc.finalize();

  FILE *fp = fopen("tmp.jpg", "wb");
  fwrite(codestream.data(), sizeof(uint8_t), codestream.size(), fp);
  fclose(fp);
  printf("Codestream length = %lld\n", codestream.size());

  // ------ DECODE
  for (size_t c = 0; c < buf.size(); ++c)
  {
    // ハフマン復号
    blkproc(buf_f[c], blk::dequantize, qtables[c > 0]); // 逆量子化
    blkproc(buf_f[c], blk::idct2);                      // 逆方向のDCT
    buf_f[c] += 128.0;
    int Hl = (YCC_HV[YCCtype][0] & 0xF0) >> 4;
    int Vl = YCC_HV[YCCtype][0] & 0x0F;
    if (c > 0)
    {
      cv::resize(buf_f[c], buf_f[c], cv::Size(), Hl, Vl, cv::INTER_AREA);
    }
    buf_f[c].convertTo(buf[c], CV_8U); // 8bitの符号なし整数に変換
  }

  cv::merge(buf, YCbCr); // inerleave形式に変換
  myYCbCr2BGR(YCbCr, recimg);

  // 画像の表示
  cv::imshow("Original", rgb);
  cv::imshow("Reconstructed", recimg);
  // キー入力を待つ 'q' で終了
  int keycode;
  do
  {
    keycode = cv::waitKey();
  } while (keycode != 'q');

  // 全てのウインドウを破棄
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}
