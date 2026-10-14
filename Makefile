ARCH ?= 80          # target SM (80=A100, 89=L40/4090, 90=H100)

.PHONY: build bench profile test cpu-test clean

build:          ## configure + build the CUDA benchmark
	cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=$(ARCH)
	cmake --build build -j

bench: build    ## run the benchmark (override: make bench ARGS="--m 8192 --fuse")
	./build/bench $(ARGS)

profile: build  ## Nsight Compute profile of the tiled kernel
	bash scripts/profile.sh

test:           ## CPU-only: build + run the reference correctness test (no GPU)
	cmake -S . -B build-cpu -DBUILD_CUDA=OFF
	cmake --build build-cpu -j
	ctest --test-dir build-cpu --output-on-failure

clean:
	rm -rf build build-cpu
