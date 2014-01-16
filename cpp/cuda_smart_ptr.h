#ifndef CUDA_SMART_PTR_H_
#define CUDA_SMART_PTR_H_

template <typename T> 
class Ptr1D {
private:
	T *ptr;
	int size_;

	Ptr1D(const Ptr1D<T> &);
	Ptr1D operator = (const Ptr1D<T> &);
public:
	
	Ptr1D(int size) 
		: size_(size)
	{
		BEGIN_FUNCTION {
			checkCudaCall(cudaMalloc((void**)&ptr, size * sizeof(T)));
		} END_FUNCTION
	}

	int size() const {
		return this->size_;
	}

	T *getDevPtr() {
		check(ptr != NULL);
		return ptr;
	}

	void free() {
		BEGIN_FUNCTION {
			checkCudaCall(cudaFree(ptr));
		} END_FUNCTION
	}
};


template <typename T> 
class Ptr2D {
private:
	T *ptr;
	int pitchElements;

	int elementsSizeX_;
	int sizeY_;

	Ptr2D(const Ptr2D<T> &);
	Ptr2D operator = (const Ptr2D<T> &);
public:
	
	Ptr2D(int elementsSizeX, int sizeY) 
		: elementsSizeX_(elementsSizeX)
		, sizeY_(sizeY)
	{
		BEGIN_FUNCTION {
			size_t pitchBytes = 0;
			checkCudaCall(cudaMallocPitch((void**)&ptr, &pitchBytes, elementsSizeX * sizeof(T), sizeY));
			pitchElements = static_cast<int>(pitchBytes / sizeof(T));
		} END_FUNCTION
	}

	int getPitchElements() const {
		return this->pitchElements;
	}

	int elementsSizeX() const {
		return this->elementsSizeX_;
	}

	int getSizeY() const {
		return this->sizeY_;
	}

	T *getDevPtr() {
		check(ptr != NULL);
		return ptr;
	}

	void free() {
		BEGIN_FUNCTION {
			checkCudaCall(cudaFree(ptr));
		} END_FUNCTION
	}
};


template <typename T>
class DeviceScopedPtr1D {
private:
	Ptr1D<T> data;
public:
	DeviceScopedPtr1D(int size) 
		: data(size)
	{
	}

	~DeviceScopedPtr1D() {
		BEGIN_DESTRUCTOR {
			data.free();
		} END_DESTRUCTOR
	}

	void copyFromHost(const T * const host, int nElements) {
		BEGIN_FUNCTION {
			check(data.size() >= nElements); 
			checkCudaCall(cudaMemcpy(data.getDevPtr(), host, nElements * sizeof(T), cudaMemcpyHostToDevice));
		} END_FUNCTION
	}

	void copyToHost(T *host, int nElements) {
		BEGIN_FUNCTION {
			check(data.size() >= nElements); 
			checkCudaCall(cudaMemcpy(host, data.getDevPtr(), nElements * sizeof(T), cudaMemcpyDeviceToHost));
		} END_FUNCTION
	}

	T* getDevPtr() {
		return data.getDevPtr();
	}
};


template <typename T>
class DeviceScopedPtr2D {
private:
	Ptr2D<T> data;
public:
	DeviceScopedPtr2D(int sizeX, int sizeY) 
		: data(sizeX, sizeY)
	{
	}

	~DeviceScopedPtr2D() {
		BEGIN_DESTRUCTOR {
			data.free();
		} END_DESTRUCTOR
	}

	void copyFromHost(const T * const host, int hostPitchElements, int elementsSizeX, int sizeY) {
		BEGIN_FUNCTION {
			check(data.elementsSizeX() >= elementsSizeX);
			
			checkCudaCall(
				cudaMemcpy2D(
					data.getDevPtr(), 
					data.getPitchElements() * sizeof(T),
					host,
					hostPitchElements * sizeof(T),
					elementsSizeX * sizeof(T), 
					sizeY,
					cudaMemcpyHostToDevice
				)
			);
			
		} END_FUNCTION
	}

	void copyToHost(T *host, int hostPitchElements, int elementsSizeX, int sizeY) {
		BEGIN_FUNCTION {
			check(data.elementsSizeX() <= elementsSizeX);
			checkCudaCall(
				cudaMemcpy2D(
					host,
					hostPitchElements * sizeof(T),
					data.getDevPtr(), 
					data.getPitchElements() * sizeof(T),
					elementsSizeX * sizeof(T), 
					sizeY,
					cudaMemcpyDeviceToHost
				);
			);
		} END_FUNCTION
	}

	T* getDevPtr() {
		return data.getDevPtr();
	}

	int getPitchElements() const {
		return data.getPitchElements();
	}
};


#endif // CUDA_SMART_PTR_H_