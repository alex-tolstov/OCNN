#include "groups.h"

Group::Group(int idx) 
	: idx(idx)
	, list(1, idx) 
	, host(idx)
{
}

void Group::rebase(std::vector<Group> &groups, int newHost) {
	for (int i = 0; i < size(); i++) {
		groups[list[i]].host = newHost;
	}
}

void Group::addAll(Group &second, std::vector<Group> &groups) {
	second.rebase(groups, this->idx);
	list.insert(list.end(), second.list.begin(), second.list.end());
}

void Group::clear() {
	list.clear();
}

int Group::size() const {
	return static_cast<int>(list.size());
}

bool GroupComparator::operator() (const Group &first, const Group &second) {
	return first.size() > second.size();
}

std::vector<Group> divideOnGroups(int nNeurons, int nIterations, float successRate, std::vector<int> &hitsHost) {
	std::vector<Group> groups;
	groups.reserve(nNeurons);
	for (int i = 0; i < nNeurons; i++) {
		groups.push_back(Group(i));
	}

	for (int i = 0; i < nNeurons; i++) {
		for (int j = 0; j < nNeurons; j++) {
			if (hitsHost[i * nNeurons + j] > successRate * nIterations) {
				if (groups[i].host != groups[j].host) {
					int host1 = groups[i].host;
					int host2 = groups[j].host;

					if (groups[host1].size() >= groups[host2].size()) {
						groups[host1].addAll(groups[host2], groups);
						groups[host2].clear();
					} else {
						groups[host2].addAll(groups[host1], groups);
						groups[host1].clear();
					}
				}
			}
		}
	}
	return groups;
}