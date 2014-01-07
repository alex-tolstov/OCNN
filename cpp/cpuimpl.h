#ifndef CPUIMPL_H_
#define CPUIMPL_H_

std::vector<int> processOscillatoryChaoticNetworkDynamicsCPU(
	int nNeurons,
	const std::vector<float> &weightMatrixHost,
	int startObservationTime,
	int nIterations,
	SyncType syncType,
	std::vector<float> &sheet,
	const float fragmentaryEPS
);

#endif // CPUIMPL_H_