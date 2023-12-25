#pragma once

#include <cmath>
#include <cstdlib>
#include <functional>
#include <opencv2/opencv.hpp>

#include "bitstream.hpp"
#include "huffman_tables.hpp"
#include "ycctype.hpp"
#include "zigzag_order.hpp"

constexpr int DCTSIZE = 8;

int clip(int v)
{
  if (v > 255)
  {
    v = 255;
  }
  if (v < 0)
  {
    v = 0;
  }
  return v;
}

void myBGR2YCbCr(cv::Mat &in, cv::Mat &out)
{
  int W = in.cols;
  int H = in.rows;
  int nc = in.channels();

  for (int y = 0; y < H; ++y)
  {
    for (int x = 0; x < nc * W; x += nc)
    {
      // BGR|BGR|BGR|BGR|...
      int idx = y * nc * W + x;
      int B = in.data[idx];
      int G = in.data[idx + 1];
      int R = in.data[idx + 2];
      int Y = 0.299 * R + 0.587 * G + 0.114 * B;
      int Cb = -0.169 * R - 0.331 * G + 0.5 * B;
      int Cr = 0.5 * R - 0.419 * G - 0.081 * B;
      out.data[idx] = clip(Y);
      out.data[idx + 1] = clip(Cb + 128);
      out.data[idx + 2] = clip(Cr + 128);
    }
  }
}

void myYCbCr2BGR(cv::Mat &in, cv::Mat &out)
{
  int W = in.cols;
  int H = in.rows;
  int nc = in.channels();

  for (int y = 0; y < H; ++y)
  {
    for (int x = 0; x < nc * W; x += nc)
    {
      // BGR|BGR|BGR|BGR|...
      int idx = y * nc * W + x;
      int Y = in.data[idx];
      int Cb = in.data[idx + 1] - 128;
      int Cr = in.data[idx + 2] - 128;

      int R = Y + 1.402 * Cr;
      int G = Y - 0.344 * Cb - 0.714 * Cr;
      int B = Y + 1.772 * Cb;
      out.data[idx] = clip(B);
      out.data[idx + 1] = clip(G);
      out.data[idx + 2] = clip(R);
    }
  }
}

void create_qtable(int c, int QF, int *qtable)
{
  if (QF == 0)
  {
    QF = 1;
  }
  float scale;
  if (QF < 50)
  {
    scale = 5000.0F / QF;
  }
  else
  {
    scale = 200.0F - 2.0F * QF;
  }
  for (int i = 0; i < DCTSIZE * DCTSIZE; ++i)
  {
    float stepsize = floorl((qmatrix[c][i] * scale + 50.0) / 100.0);
    if (stepsize < 1.0)
    {
      stepsize = 1.0;
    }
    if (stepsize > 255.0)
    {
      stepsize = 255.0;
    }
    qtable[i] = stepsize;
  }
}

void blkproc(cv::Mat &in, std::function<void(cv::Mat &, int *)> func,
             int *p = nullptr)
{
  for (int y = 0; y < in.rows; y += DCTSIZE)
  {
    for (int x = 0; x < in.cols; x += DCTSIZE)
    {
      cv::Mat blk_in, blk_out;
      blk_in = in(cv::Rect(x, y, DCTSIZE, DCTSIZE)).clone();
      blk_out = in(cv::Rect(x, y, DCTSIZE, DCTSIZE));
      func(blk_in, p);
      blk_in.convertTo(blk_out, blk_out.type());
    }
  }
}

static int ilog2(uint32_t x)
{
  int s = 0;
  int bound = 1;
  while (x >= bound)
  {
    bound += bound;
    s++;
  }
  return s;
}

static void EncodeDC(int diff, const uint32_t *Ctable, const uint32_t *Ltable,
                     bitstream &enc)
{
  int uval = abs(diff);
  int s = ilog2(uval);
  enc.put_bits(Ctable[s], Ltable[s]);
  if (s != 0)
  {
    if (diff < 0)
    {
      diff -= 1;
    }
    enc.put_bits(diff, s);
  }
}

static void EncodeAC(int run, int val, const uint32_t *Ctable,
                     const uint32_t *Ltable, bitstream &enc)
{
  int uval = abs(val);
  int s = ilog2(uval);
  enc.put_bits(Ctable[(run << 4) + s], Ltable[(run << 4) + s]);
  if (s != 0)
  {
    if (val < 0)
    {
      val -= 1;
    }
    enc.put_bits(val, s);
  }
}

static void encode_block(cv::Mat &in, int c, int &prev_dc, bitstream &enc)
{
  float *p = (float *)in.data;
  int diff = p[0] - prev_dc;
  prev_dc = p[0];
  EncodeDC(diff, DC_cwd[c], DC_len[c], enc);

  int ac, run = 0;
  for (int i = 1; i < 64; ++i)
  {
    ac = p[scan[i]];
    if (ac == 0)
    {
      run++;
    }
    else
    {
      while (run > 15)
      {
        // ZRL
        EncodeAC(0xF, 0x0, AC_cwd[c], AC_len[c], enc);
        run -= 16;
      }
      EncodeAC(run, ac, AC_cwd[c], AC_len[c], enc);
      run = 0;
    }
  }
  if (run)
  {
    // EOB
    EncodeAC(0x0, 0x0, AC_cwd[c], AC_len[c], enc);
  }
}

void entropy_coding(std::vector<cv::Mat> &in, int YCCtype, bitstream &enc)
{
  const int ncomp = in.size(); // 1 or 3
  int prev_dc[3] = {0};
  cv::Mat blk;

  int Hl = (YCC_HV[YCCtype][0] & 0xF0) >> 4;
  int Vl = YCC_HV[YCCtype][0] & 0x0F;
  for (int y = 0, cy = 0; y < in[0].rows; y += DCTSIZE * Vl, cy += DCTSIZE)
  {
    for (int x = 0, cx = 0; x < in[0].cols; x += DCTSIZE * Hl, cx += DCTSIZE)
    {
      for (int ny = 0; ny < Vl; ++ny)
      {
        for (int nx = 0; nx < Hl; ++nx)
        {
          blk = in[0](cv::Rect(x + nx * DCTSIZE, y + ny * DCTSIZE, DCTSIZE, DCTSIZE)).clone();
          encode_block(blk, 0, prev_dc[0], enc);
        }
      }
      if (ncomp > 1)
      {
        blk = in[1](cv::Rect(cx, cy, DCTSIZE, DCTSIZE)).clone();
        encode_block(blk, 1, prev_dc[1], enc);
        blk = in[2](cv::Rect(cx, cy, DCTSIZE, DCTSIZE)).clone();
        encode_block(blk, 1, prev_dc[2], enc);
      }
    }
  }
}

namespace blk
{
  void dct2(cv::Mat &in, int *dummy)
  {
    if (in.rows != DCTSIZE || in.cols != DCTSIZE || in.channels() != 1)
    {
      printf("input for block_dct2() shall be 8x8 and monochrome.\n");
      exit(EXIT_FAILURE);
    }
    cv::dct(in, in);
  }

  void idct2(cv::Mat &in, int *dummry)
  {
    if (in.rows != DCTSIZE || in.cols != DCTSIZE || in.channels() != 1)
    {
      printf("input for block_idct2() shall be 8x8 and monochrome.\n");
      exit(EXIT_FAILURE);
    }
    cv::idct(in, in);
  }

  void quantize(cv::Mat &in, int *qtable)
  {
    float *coeff = (float *)in.data;
    for (int i = 0; i < DCTSIZE * DCTSIZE; ++i)
    {
      float x = coeff[i];
      float sign = (x >= 0) ? 1 : -1;
      x = fabs(x);
      x /= qtable[i];
      x += 0.5;
      x = floorl(x);
      coeff[i] = sign * x;
    }
  }

  void dequantize(cv::Mat &in, int *qtable)
  {
    float *coeff = (float *)in.data;
    for (int i = 0; i < DCTSIZE * DCTSIZE; ++i)
    {
      float x = coeff[i];
      float sign = (x >= 0) ? 1 : -1;
      x = fabs(x);
      x *= qtable[i];
      x += 0.5;
      x = floorl(x);
      coeff[i] = sign * x;
    }
  }
} // end of namespace blk