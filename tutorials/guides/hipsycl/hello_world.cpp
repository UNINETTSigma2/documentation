#include <SYCL/sycl.hpp>

int main (int argc, char** argv) {
  auto Q = sycl::queue{sycl::default_selector{}};

  std::cout << "Chosen device: "
            << Q.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  return EXIT_SUCCESS;
}
