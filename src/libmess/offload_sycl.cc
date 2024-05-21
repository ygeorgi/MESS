#include "offload.hh"
#include "io.hh"
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

Lapack::Vector Offload::Sycl::eigenvalues (Lapack::SymmetricMatrix m, Lapack::Matrix* evec)
{  
  const char funame [] = "Offload::Sycl::eigenvalues: ";

  //using namespace oneapi::mkl;
  
  if(!m.isinit()) {
    //
    std::cerr << funame << "not initialized\n";
    
    throw Error::Init();
  }

  Lapack::Matrix a = m;

  Lapack::Vector res(m.size());

  oneapi::mkl::uplo uplo = oneapi::mkl::uplo::upper;
  
  oneapi::mkl::job  jobz = oneapi::mkl::job::N;

  if(evec) {
    //
    jobz = oneapi::mkl::job::V;
  }
  
  std::int64_t n = m.size();

  std::int64_t nn = n * n;

  // compute on device
  //
  {
    sycl::queue queue(sycl::gpu_selector_v);

    IO::log << IO::log_offset << funame << "running on: "
	    << queue.get_device().get_info<sycl::info::device::name>()
	    << std::endl;

    auto lwork = oneapi::mkl::lapack::syevd_scratchpad_size<double>(queue, jobz, uplo, n, n);

    auto work          = sycl::malloc_device<double>(lwork, queue);

    auto res_on_device = sycl::malloc_device<double>(n, queue);
  
    auto   a_on_device = sycl::malloc_device<double>(nn, queue);

    queue.copy((const double*)a, a_on_device, nn).wait_and_throw();

    oneapi::mkl::lapack::syevd(queue, jobz, uplo, n, a_on_device, n,
			       //
			       res_on_device, work, lwork).wait_and_throw();

    queue.copy(res_on_device, (double*)res, n);

    if(evec)
      //
      queue.copy(a_on_device, (double*)*evec, nn);

    queue.wait_and_throw();

    sycl::free(a_on_device,   queue);

    sycl::free(res_on_device, queue);

    sycl::free(work,          queue);
  }
  
  return res;
}
