#ifndef __IMAGEM_HPP
#define __IMAGEM_HPP

#include <fstream>
#include <string>

class Imagem {
    public:
    
    int rows, cols, total_size;
    unsigned char *pixels;


    Imagem(int rows, int cols) : rows(rows), cols(cols) {
        total_size = rows * cols;
        pixels = new unsigned char[total_size];
    }

    static Imagem read(std::string path) {
        std::ifstream inf(path);
        std::string first_line;
        std::getline(inf, first_line);
        if (first_line != "P2") return Imagem(0, 0);

        int rows, cols;
        inf >> cols;
        inf >> rows;

        Imagem img(rows, cols);
        int temp;
        inf >> temp;
        
        for (int k = 0; k < img.total_size; k++) {
            int t;
            inf >> t;
            img.pixels[k] = t;
        }

        return img;
    }

    void write(std::string path) {
        std::ofstream of(path);
        of << "P2\n" << cols << " " <<  rows << " 255\n";
        for (int k = 0; k < total_size; k++) {
            of << ((int) pixels[k]) << " ";
        }
    }
};

#endif