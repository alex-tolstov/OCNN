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


#endif // CUDA_SMART_PTR_H_