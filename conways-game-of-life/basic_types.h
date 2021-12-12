#pragma once

#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/core/cs.hpp>

namespace bg = boost::geometry;
namespace bc = bg::cs;
namespace bm = bg::model;

typedef bm::point<int64_t, 2, bc::cartesian> Point;



inline bool operator== (const Point& lhs, const Point& rhs)
{
	return lhs.get<0>() == rhs.get<0>() && lhs.get<1>() == rhs.get<1>();
}

inline bool operator< (const Point& lhs, const Point& rhs)
{
	bool c = lhs.get<0>() < rhs.get<0>();
	if (c) return c;
	return lhs.get<0>() == rhs.get<0>() ? lhs.get<1>() < rhs.get<1>() : false;
}

struct CrutchPredLess
{
	inline bool operator() (const Point& lhs, const Point& rhs) const
	{
		return lhs < rhs;
	}
	inline bool operator() (Point& lhs, Point& rhs) const
	{
		return lhs < rhs;
	}
	inline bool operator() (Point& lhs, Point& rhs) 
	{
		return lhs < rhs;
	}
};