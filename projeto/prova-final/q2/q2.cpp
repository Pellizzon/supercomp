#include <iostream>
#include <chrono>
#include <unistd.h>

void func1() {
    sleep(2);
}

void func2() {
    sleep(7);
}

void func3() {
    sleep(3);
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    func1();
    func2();
    func3();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << duration.count() << "s\n";

    return 0;
}