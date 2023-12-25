#pragma once
// #include <string>
// #include <vector>

// clang-format off
constexpr unsigned int DC_len[2][16] = {
    {2, 3, 3, 3, 3, 3, 4, 5, 6, 7,  8,  9, 0, 0, 0, 0}, //10, 11, 12, 13},
    {2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0}  //12, 13, 14, 15},
};

constexpr unsigned int DC_cwd[2][16] = {
    {0x0000, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x000e, 0x001e, 0x003e, 0x007e, 0x00fe, 0x01fe, 0, 0, 0, 0}, // 0x03fe, 0x07fe, 0x0ffe, 0x1ffe},
    {0x0000, 0x0001, 0x0002, 0x0006, 0x000e, 0x001e, 0x003e, 0x007e, 0x00fe, 0x01fe, 0x03fe, 0x07fe, 0, 0, 0, 0}  // 0x0ffe, 0x1ffe, 0x3ffe, 0x7ffe}
};

constexpr unsigned int AC_len[2][256] = {
    { 4,  2,  2,  3,  4,  5,  7,  8, 10, 16, 16, 0, 0, 0, 0, 0,  
      0,  4,  5,  7,  9, 11, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,    
      0,  5,  8, 10, 12, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,    
      0,  6,  9, 12, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,    
      0,  6, 10, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,
      0,  7, 11, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,
      0,  7, 12, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,
      0,  8, 12, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,
      0,  9, 15, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,
      0,  9, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,
      0,  9, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,
      0, 10, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,
      0, 10, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,
      0, 11, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,
      0, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,
     11, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0},
    { 2,  2,  3,  4,  5,  5,  6,  7,  9, 10, 12, 0, 0, 0, 0, 0,
      0,  4,  6,  8,  9, 11, 12, 16, 16, 16, 16, 0, 0, 0, 0, 0,  
      0,  5,  8, 10, 12, 15, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,  
      0,  5,  8, 10, 12, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,  
      0,  6,  9, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,  
      0,  6, 10, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,  
      0,  7, 11, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,  
      0,  7, 11, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,  
      0,  8, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,  
      0,  9, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,  
      0,  9, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,  
      0,  9, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,  
      0,  9, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,
      0, 11, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,  
      0, 14, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0,
     10, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0}
};

constexpr unsigned int AC_cwd[2][256] = {
    {
     0x000a, 0x0000, 0x0001, 0x0004, 0x000b, 0x001a, 0x0078, 0x00f8, 0x03f6, 0xff82, 0xff83, 0, 0, 0, 0, 0,
     0x0000, 0x000c, 0x001b, 0x0079, 0x01f6, 0x07f6, 0xff84, 0xff85, 0xff86, 0xff87, 0xff88, 0, 0, 0, 0, 0,
     0x0000, 0x001c, 0x00f9, 0x03f7, 0x0ff4, 0xff89, 0xff8a, 0xff8b, 0xff8c, 0xff8d, 0xff8e, 0, 0, 0, 0, 0,
     0x0000, 0x003a, 0x01f7, 0x0ff5, 0xff8f, 0xff90, 0xff91, 0xff92, 0xff93, 0xff94, 0xff95, 0, 0, 0, 0, 0,
     0x0000, 0x003b, 0x03f8, 0xff96, 0xff97, 0xff98, 0xff99, 0xff9a, 0xff9b, 0xff9c, 0xff9d, 0, 0, 0, 0, 0,
     0x0000, 0x007a, 0x07f7, 0xff9e, 0xff9f, 0xffa0, 0xffa1, 0xffa2, 0xffa3, 0xffa4, 0xffa5, 0, 0, 0, 0, 0,
     0x0000, 0x007b, 0x0ff6, 0xffa6, 0xffa7, 0xffa8, 0xffa9, 0xffaa, 0xffab, 0xffac, 0xffad, 0, 0, 0, 0, 0,
     0x0000, 0x00fa, 0x0ff7, 0xffae, 0xffaf, 0xffb0, 0xffb1, 0xffb2, 0xffb3, 0xffb4, 0xffb5, 0, 0, 0, 0, 0,
     0x0000, 0x01f8, 0x7fc0, 0xffb6, 0xffb7, 0xffb8, 0xffb9, 0xffba, 0xffbb, 0xffbc, 0xffbd, 0, 0, 0, 0, 0,
     0x0000, 0x01f9, 0xffbe, 0xffbf, 0xffc0, 0xffc1, 0xffc2, 0xffc3, 0xffc4, 0xffc5, 0xffc6, 0, 0, 0, 0, 0,
     0x0000, 0x01fa, 0xffc7, 0xffc8, 0xffc9, 0xffca, 0xffcb, 0xffcc, 0xffcd, 0xffce, 0xffcf, 0, 0, 0, 0, 0,
     0x0000, 0x03f9, 0xffd0, 0xffd1, 0xffd2, 0xffd3, 0xffd4, 0xffd5, 0xffd6, 0xffd7, 0xffd8, 0, 0, 0, 0, 0,
     0x0000, 0x03fa, 0xffd9, 0xffda, 0xffdb, 0xffdc, 0xffdd, 0xffde, 0xffdf, 0xffe0, 0xffe1, 0, 0, 0, 0, 0,
     0x0000, 0x07f8, 0xffe2, 0xffe3, 0xffe4, 0xffe5, 0xffe6, 0xffe7, 0xffe8, 0xffe9, 0xffea, 0, 0, 0, 0, 0,
     0x0000, 0xffeb, 0xffec, 0xffed, 0xffee, 0xffef, 0xfff0, 0xfff1, 0xfff2, 0xfff3, 0xfff4, 0, 0, 0, 0, 0,
     0x07f9, 0xfff5, 0xfff6, 0xfff7, 0xfff8, 0xfff9, 0xfffa, 0xfffb, 0xfffc, 0xfffd, 0xfffe, 0, 0, 0, 0, 0
    },
    {
     0x0000, 0x0001, 0x0004, 0x000a, 0x0018, 0x0019, 0x0038, 0x0078, 0x01f4, 0x03f6, 0x0ff4, 0, 0, 0, 0, 0,
     0x0000, 0x000b, 0x0039, 0x00f6, 0x01f5, 0x07f6, 0x0ff5, 0xff88, 0xff89, 0xff8a, 0xff8b, 0, 0, 0, 0, 0,
     0x0000, 0x001a, 0x00f7, 0x03f7, 0x0ff6, 0x7fc2, 0xff8c, 0xff8d, 0xff8e, 0xff8f, 0xff90, 0, 0, 0, 0, 0,
     0x0000, 0x001b, 0x00f8, 0x03f8, 0x0ff7, 0xff91, 0xff92, 0xff93, 0xff94, 0xff95, 0xff96, 0, 0, 0, 0, 0,
     0x0000, 0x003a, 0x01f6, 0xff97, 0xff98, 0xff99, 0xff9a, 0xff9b, 0xff9c, 0xff9d, 0xff9e, 0, 0, 0, 0, 0,
     0x0000, 0x003b, 0x03f9, 0xff9f, 0xffa0, 0xffa1, 0xffa2, 0xffa3, 0xffa4, 0xffa5, 0xffa6, 0, 0, 0, 0, 0,
     0x0000, 0x0079, 0x07f7, 0xffa7, 0xffa8, 0xffa9, 0xffaa, 0xffab, 0xffac, 0xffad, 0xffae, 0, 0, 0, 0, 0,
     0x0000, 0x007a, 0x07f8, 0xffaf, 0xffb0, 0xffb1, 0xffb2, 0xffb3, 0xffb4, 0xffb5, 0xffb6, 0, 0, 0, 0, 0,
     0x0000, 0x00f9, 0xffb7, 0xffb8, 0xffb9, 0xffba, 0xffbb, 0xffbc, 0xffbd, 0xffbe, 0xffbf, 0, 0, 0, 0, 0,
     0x0000, 0x01f7, 0xffc0, 0xffc1, 0xffc2, 0xffc3, 0xffc4, 0xffc5, 0xffc6, 0xffc7, 0xffc8, 0, 0, 0, 0, 0,
     0x0000, 0x01f8, 0xffc9, 0xffca, 0xffcb, 0xffcc, 0xffcd, 0xffce, 0xffcf, 0xffd0, 0xffd1, 0, 0, 0, 0, 0,
     0x0000, 0x01f9, 0xffd2, 0xffd3, 0xffd4, 0xffd5, 0xffd6, 0xffd7, 0xffd8, 0xffd9, 0xffda, 0, 0, 0, 0, 0,
     0x0000, 0x01fa, 0xffdb, 0xffdc, 0xffdd, 0xffde, 0xffdf, 0xffe0, 0xffe1, 0xffe2, 0xffe3, 0, 0, 0, 0, 0,
     0x0000, 0x07f9, 0xffe4, 0xffe5, 0xffe6, 0xffe7, 0xffe8, 0xffe9, 0xffea, 0xffeb, 0xffec, 0, 0, 0, 0, 0,
     0x0000, 0x3fe0, 0xffed, 0xffee, 0xffef, 0xfff0, 0xfff1, 0xfff2, 0xfff3, 0xfff4, 0xfff5, 0, 0, 0, 0, 0,
     0x03fa, 0x7fc3, 0xfff6, 0xfff7, 0xfff8, 0xfff9, 0xfffa, 0xfffb, 0xfffc, 0xfffd, 0xfffe, 0, 0, 0, 0, 0
    }
};
// clang-format on
// Table: K-5: Luminance AC coefficients
// const std::vector<std::vector<std::string>> AC_LUMA_HUFF_CODES = {
//     {
//         "1010",              // 0/0 (EOB)
//         "00",                // 0/1
//         "01",                // 0/2
//         "100",               // 0/3
//         "1011",              // 0/4
//         "11010",             // 0/5
//         "1111000",           // 0/6
//         "11111000",          // 0/7
//         "1111110110",        // 0/8
//         "1111111110000010",  // 0/9
//         "1111111110000011"   // 0/A
//     },

//     {
//         "",                  // 1/0
//         "1100",              // 1/1
//         "11011",             // 1/2
//         "1111001",           // 1/3
//         "111110110",         // 1/4
//         "11111110110",       // 1/5
//         "1111111110000100",  // 1/6
//         "1111111110000101",  // 1/7
//         "1111111110000110",  // 1/8
//         "1111111110000111",  // 1/9
//         "1111111110001000"   // 1/A
//     },

//     {
//         "",                  // 2/0
//         "11100",             // 2/1
//         "11111001",          // 2/2
//         "1111110111",        // 2/3
//         "111111110100",      // 2/4
//         "1111111110001001",  // 2/5
//         "1111111110001010",  // 2/6
//         "1111111110001011",  // 2/7
//         "1111111110001100",  // 2/8
//         "1111111110001101",  // 2/9
//         "1111111110001110"   // 2/A
//     },

//     {
//         "",                  // 3/0
//         "111010",            // 3/1
//         "111110111",         // 3/2
//         "111111110101",      // 3/3
//         "1111111110001111",  // 3/4
//         "1111111110010000",  // 3/5
//         "1111111110010001",  // 3/6
//         "1111111110010010",  // 3/7
//         "1111111110010011",  // 3/8
//         "1111111110010100",  // 3/9
//         "1111111110010101"   // 3/A
//     },

//     {
//         "",                  // 4/0
//         "111011",            // 4/1
//         "1111111000",        // 4/2
//         "1111111110010110",  // 4/3
//         "1111111110010111",  // 4/4
//         "1111111110011000",  // 4/5
//         "1111111110011001",  // 4/6
//         "1111111110011010",  // 4/7
//         "1111111110011011",  // 4/8
//         "1111111110011100",  // 4/9
//         "1111111110011101"   // 4/A
//     },

//     {
//         "",                  // 5/0
//         "1111010",           // 5/1
//         "11111110111",       // 5/2
//         "1111111110011110",  // 5/3
//         "1111111110011111",  // 5/4
//         "1111111110100000",  // 5/5
//         "1111111110100001",  // 5/6
//         "1111111110100010",  // 5/7
//         "1111111110100011",  // 5/8
//         "1111111110100100",  // 5/9
//         "1111111110100101"   // 5/A
//     },

//     {
//         "",                  // 6/0
//         "1111011",           // 6/1
//         "111111110110",      // 6/2
//         "1111111110100110",  // 6/3
//         "1111111110100111",  // 6/4
//         "1111111110101000",  // 6/5
//         "1111111110101001",  // 6/6
//         "1111111110101010",  // 6/7
//         "1111111110101011",  // 6/8
//         "1111111110101100",  // 6/9
//         "1111111110101101"   // 6/A
//     },

//     {
//         "",                  // 7/0
//         "11111010",          // 7/1
//         "111111110111",      // 7/2
//         "1111111110101110",  // 7/3
//         "1111111110101111",  // 7/4
//         "1111111110110000",  // 7/5
//         "1111111110110001",  // 7/6
//         "1111111110110010",  // 7/7
//         "1111111110110011",  // 7/8
//         "1111111110110100",  // 7/9
//         "1111111110110101"   // 7/A
//     },

//     {
//         "",                  // 8/0
//         "111111000",         // 8/1
//         "111111111000000",   // 8/2
//         "1111111110110110",  // 8/3
//         "1111111110110111",  // 8/4
//         "1111111110111000",  // 8/5
//         "1111111110111001",  // 8/6
//         "1111111110111010",  // 8/7
//         "1111111110111011",  // 8/8
//         "1111111110111100",  // 8/9
//         "1111111110111101"   // 8/A
//     },
//     {"",                   // 9/0
//      "111111001",          // 9/1
//      "1111111110111110",   // 9/2
//      "1111111110111111",   // 9/3
//      "1111111111000000",   // 9/4
//      "1111111111000001",   // 9/5
//      "1111111111000010",   // 9/6
//      "1111111111000011",   // 9/7
//      "1111111111000100",   // 9/8
//      "1111111111000101",   // 9/9
//      "1111111111000110"},  // 9/A
//     {"",                   // A/0
//      "111111010",          // A/1
//      "1111111111000111",   // A/2
//      "1111111111001000",   // A/3
//      "1111111111001001",   // A/4
//      "1111111111001010",   // A/5
//      "1111111111001011",   // A/6
//      "1111111111001100",   // A/7
//      "1111111111001101",   // A/8
//      "1111111111001110",   // A/9
//      "1111111111001111"},  // A/A
//     {"",                   // B/0
//      "1111111001",         // B/1
//      "1111111111010000",   // B/2
//      "1111111111010001",   // B/3
//      "1111111111010010",   // B/4
//      "1111111111010011",   // B/5
//      "1111111111010100",   // B/6
//      "1111111111010101",   // B/7
//      "1111111111010110",   // B/8
//      "1111111111010111",   // B/9
//      "1111111111011000"},  // B/A
//     {"",                   // C/0
//      "1111111010",         // C/1
//      "1111111111011001",   // C/2
//      "1111111111011010",   // C/3
//      "1111111111011011",   // C/4
//      "1111111111011100",   // C/5
//      "1111111111011101",   // C/6
//      "1111111111011110",   // C/7
//      "1111111111011111",   // C/8
//      "1111111111100000",   // C/9
//      "1111111111100001"},  // C/A
//     {"",                   // D/0
//      "11111111000",        // D/1
//      "1111111111100010",   // D/2
//      "1111111111100011",   // D/3
//      "1111111111100100",   // D/4
//      "1111111111100101",   // D/5
//      "1111111111100110",   // D/6
//      "1111111111100111",   // D/7
//      "1111111111101000",   // D/8
//      "1111111111101001",   // D/9
//      "1111111111101010"},  // D/A
//     {"",                   // E/0
//      "1111111111101011",   // E/1
//      "1111111111101100",   // E/2
//      "1111111111101101",   // E/3
//      "1111111111101110",   // E/4
//      "1111111111101111",   // E/5
//      "1111111111110000",   // E/6
//      "1111111111110001",   // E/7
//      "1111111111110010",   // E/8
//      "1111111111110011",   // E/9
//      "1111111111110100"},  // E/A
//     {"11111111001",        // F/0 (Zero-run length)
//      "1111111111110101",   // F/1
//      "1111111111110110",   // F/2
//      "1111111111110111",   // F/3
//      "1111111111111000",   // F/4
//      "1111111111111001",   // F/5
//      "1111111111111010",   // F/6
//      "1111111111111011",   // F/7
//      "1111111111111100",   // F/8
//      "1111111111111101",   // F/9
//      "1111111111111110"}   // F/A
// };

// // Table: K-5: Chrominance AC coefficients
// const std::vector<std::vector<std::string>> AC_CHROMA_HUFF_CODES = {
//     {"00",                 // 0/0 (EOB)
//      "01",                 // 0/1
//      "100",                // 0/2
//      "1010",               // 0/3
//      "11000",              // 0/4
//      "11001",              // 0/5
//      "111000",             // 0/6
//      "1111000",            // 0/7
//      "111110100",          // 0/8
//      "1111110110",         // 0/9
//      "111111110100"},      // 0/A
//     {"",                   // 1/0
//      "1011",               // 1/1
//      "111001",             // 1/2
//      "11110110",           // 1/3
//      "111110101",          // 1/4
//      "11111110110",        // 1/5
//      "111111110101",       // 1/6
//      "1111111110001000",   // 1/7
//      "1111111110001001",   // 1/8
//      "1111111110001010",   // 1/9
//      "1111111110001011"},  // 1/A
//     {"",                   // 2/0
//      "11010",              // 2/1
//      "11110111",           // 2/2
//      "1111110111",         // 2/3
//      "111111110110",       // 2/4
//      "111111111000010",    // 2/5
//      "1111111110001100",   // 2/6
//      "1111111110001101",   // 2/7
//      "1111111110001110",   // 2/8
//      "1111111110001111",   // 2/9
//      "1111111110010000"},  // 2/A
//     {"",                   // 3/0
//      "11011",              // 3/1
//      "11111000",           // 3/2
//      "1111111000",         // 3/3
//      "111111110111",       // 3/4
//      "1111111110010001",   // 3/5
//      "1111111110010010",   // 3/6
//      "1111111110010011",   // 3/7
//      "1111111110010100",   // 3/8
//      "1111111110010101",   // 3/9
//      "1111111110010110"},  // 3/A
//     {"",                   // 4/0
//      "111010",             // 4/1
//      "111110110",          // 4/2
//      "1111111110010111",   // 4/3
//      "1111111110011000",   // 4/4
//      "1111111110011001",   // 4/5
//      "1111111110011010",   // 4/6
//      "1111111110011011",   // 4/7
//      "1111111110011100",   // 4/8
//      "1111111110011101",   // 4/9
//      "1111111110011110"},  // 4/A
//     {"",                   // 5/0
//      "111011",             // 5/1
//      "1111111001",         // 5/2
//      "1111111110011111",   // 5/3
//      "1111111110100000",   // 5/4
//      "1111111110100001",   // 5/5
//      "1111111110100010",   // 5/6
//      "1111111110100011",   // 5/7
//      "1111111110100100",   // 5/8
//      "1111111110100101",   // 5/9
//      "1111111110100110"},  // 5/A
//     {"",                   // 6/0
//      "1111001",            // 6/1
//      "11111110111",        // 6/2
//      "1111111110100111",   // 6/3
//      "1111111110101000",   // 6/4
//      "1111111110101001",   // 6/5
//      "1111111110101010",   // 6/6
//      "1111111110101011",   // 6/7
//      "1111111110101100",   // 6/8
//      "1111111110101101",   // 6/9
//      "1111111110101110"},  // 6/A
//     {"",                   // 7/0
//      "1111010",            // 7/1
//      "11111111000",        // 7/2
//      "1111111110101111",   // 7/3
//      "1111111110110000",   // 7/4
//      "1111111110110001",   // 7/5
//      "1111111110110010",   // 7/6
//      "1111111110110011",   // 7/7
//      "1111111110110100",   // 7/8
//      "1111111110110101",   // 7/9
//      "1111111110110110"},  // 7/A
//     {"",                   // 8/0
//      "11111001",           // 8/1
//      "1111111110110111",   // 8/2
//      "1111111110111000",   // 8/3
//      "1111111110111001",   // 8/4
//      "1111111110111010",   // 8/5
//      "1111111110111011",   // 8/6
//      "1111111110111100",   // 8/7
//      "1111111110111101",   // 8/8
//      "1111111110111110",   // 8/9
//      "1111111110111111"},  // 8/A
//     {"",                   // 9/0
//      "111110111",          // 9/1
//      "1111111111000000",   // 9/2
//      "1111111111000001",   // 9/3
//      "1111111111000010",   // 9/4
//      "1111111111000011",   // 9/5
//      "1111111111000100",   // 9/6
//      "1111111111000101",   // 9/7
//      "1111111111000110",   // 9/8
//      "1111111111000111",   // 9/9
//      "1111111111001000"},  // 9/A
//     {"",                   // A/0
//      "111111000",          // A/1
//      "1111111111001001",   // A/2
//      "1111111111001010",   // A/3
//      "1111111111001011",   // A/4
//      "1111111111001100",   // A/5
//      "1111111111001101",   // A/6
//      "1111111111001110",   // A/7
//      "1111111111001111",   // A/8
//      "1111111111010000",   // A/9
//      "1111111111010001"},  // A/A
//     {"",                   // B/0
//      "111111001",          // B/1
//      "1111111111010010",   // B/2
//      "1111111111010011",   // B/3
//      "1111111111010100",   // B/4
//      "1111111111010101",   // B/5
//      "1111111111010110",   // B/6
//      "1111111111010111",   // B/7
//      "1111111111011000",   // B/8
//      "1111111111011001",   // B/9
//      "1111111111011010"},  // B/A
//     {"",                   // C/0
//      "111111010",          // C/1
//      "1111111111011011",   // C/2
//      "1111111111011100",   // C/3
//      "1111111111011101",   // C/4
//      "1111111111011110",   // C/5
//      "1111111111011111",   // C/6
//      "1111111111100000",   // C/7
//      "1111111111100001",   // C/8
//      "1111111111100010",   // C/9
//      "1111111111100011"},  // C/A
//     {"",                   // D/0
//      "11111111001",        // D/1
//      "1111111111100100",   // D/2
//      "1111111111100101",   // D/3
//      "1111111111100110",   // D/4
//      "1111111111100111",   // D/5
//      "1111111111101000",   // D/6
//      "1111111111101001",   // D/7
//      "1111111111101010",   // D/8
//      "1111111111101011",   // D/9
//      "1111111111101100"},  // D/A
//     {"",                   // E/0
//      "11111111100000",     // E/1
//      "1111111111101101",   // E/2
//      "1111111111101110",   // E/3
//      "1111111111101111",   // E/4
//      "1111111111110000",   // E/5
//      "1111111111110001",   // E/6
//      "1111111111110010",   // E/7
//      "1111111111110011",   // E/8
//      "1111111111110100",   // E/9
//      "1111111111110101"},  // E/A
//     {"1111111010",         // F/0 (Zero-run length)
//      "111111111000011",    // F/1
//      "1111111111110110",   // F/2
//      "1111111111110111",   // F/3
//      "1111111111111000",   // F/4
//      "1111111111111001",   // F/5
//      "1111111111111010",   // F/6
//      "1111111111111011",   // F/7
//      "1111111111111100",   // F/8
//      "1111111111111101",   // F/9
//      "1111111111111110"}   // F/A
// };