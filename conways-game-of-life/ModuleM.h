#pragma once

#include "basic_types.h"
#include "constants.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <xmemory>
#include <vector>
#include <set>

class GameField
{
public:
	GameField(size_t w, size_t h);
	GameField(Point p);
	~GameField();

	bool* data();
	const bool* const data() const;
	bool& operator[] (const Point& p);
	const bool& operator[] (const Point& p) const;
	const Point& getMaxSize() const;
private:
	Point maxSize;
	bool* gameFieldData;

	inline size_t getIdx(const Point& p) const;
	void allocMemForGameField();
};

class Data
{
public:
	Data() : field{MAX_W, MAX_H} {}

	bool changeCell(Point cell);
private:
	GameField field;
	std::set<Point, CrutchPredLess> aliveCells;

	friend class GameTicker;
	friend class GameViewer;
};