#ifndef GROUPS_H_
#define GROUPS_H_

#include <vector>
#include <functional>

struct Group {

	int idx;

	int host;
	
	std::vector<int> list;

	Group(int idx);

	void rebase(std::vector<Group> &groups, int newHost);

	void addAll(Group &second, std::vector<Group> &groups);

	void clear();

	int size() const;
};

struct GroupComparator : public std::binary_function<Group, Group, bool> {
public:
	bool operator() (const Group &first, const Group &second);
};

std::vector<Group> divideOnGroups(int nNeurons, int nIterations, float successRate, std::vector<int> &hitsHost);

#endif // GROUPS_H_