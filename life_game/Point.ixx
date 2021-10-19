export module Point;

import <math.h>;
import <cstdint>;

export namespace pnd
{

	struct Point2
	{
		int32_t x;
		int32_t y;
		float w;
	};

	struct Point3
	{
		int32_t x;
		int32_t y;
		int32_t z;
		float w;
	};

	struct Vector2
	{
		int32_t x;
		int32_t y;
		float w;
	};

	struct Vector3
	{
		int32_t x;
		int32_t y;
		int32_t z;
		float w;
	};



	constexpr float for_rounding = 0.5;

	template<typename _Ty>
	_Ty scale2(Point2* origin, _Ty* src, float scale_factor)
	{
		_Ty res;
		float tmp_scale = src->w * scale_factor < 1 ? 1 / src->w : scale_factor;

		res.x = (uint32_t)((src->x - origin->x) * tmp_scale);
		res.y = (uint32_t)((src->y - origin->y) * tmp_scale);
		res.w = src->w * tmp_scale;

		return res;
	}

	template<typename _Ty>
	_Ty scale3(Point3* origin, _Ty* src, float scale_factor)
	{
		_Ty res;
		float tmp_scale = src->w * scale_factor < 1 ? 1 / src->w : scale_factor;

		res.x = (uint32_t)((src->x - origin->x) * tmp_scale);
		res.y = (uint32_t)((src->y - origin->y) * tmp_scale);
		res.z = (uint32_t)((src->z - origin->z) * tmp_scale);
		res.w = src->w * tmp_scale;

		return res;
	}
	


	template<typename _Ty>
	_Ty rotateXY_2(Point2* origin, _Ty* src, float radians)
	{
		float _sin = sinf(radians);
		float _cos = cosf(radians);
		float rounding;
		float first;

		_Ty res;

		int x = src->x - origin->x;
		int y = src->y - origin->y;

		first = _cos * x - _sin * y;
		rounding = for_rounding * (first > 0 ? +1 : -1);
		res.x = (int32_t)(first + rounding);

		first = _sin * x + _cos * y;
		rounding = for_rounding * (first > 0 ? +1 : -1);
		res.y = (int32_t)(first + rounding);

		res.w = src->w;
		
		return res;
	}

	template<typename _Ty>
	_Ty rotateXY_3(Point3* origin, _Ty* src, float radians)
	{
		float _sin = sinf(radians);
		float _cos = cosf(radians);
		float rounding;
		float first;

		_Ty res;

		int x = src->x - origin->x;
		int y = src->y - origin->y;
		int z = src->z - origin->z;

		first = _cos * x - _sin * y;
		rounding = for_rounding * (first > 0 ? +1 : -1);
		res.x = (int32_t)(first + rounding);

		first = _sin * x + _cos * y;
		rounding = for_rounding * (first > 0 ? +1 : -1);
		res.y = (int32_t)(first + rounding);


		res.z = z;
		res.w = src->w;

		return res;
	}

	template<typename _Ty>
	_Ty rotateYZ_3(Point3* origin, _Ty* src, float radians)
	{
		float _sin = sinf(radians);
		float _cos = cosf(radians);
		float first;
		float rounding;

		_Ty res;

		int x = src->x - origin->x;
		int y = src->y - origin->y;
		int z = src->z - origin->z;

		first = _cos * x + _sin * z;
		rounding = for_rounding * (first > 0 ? +1 : -1);
		res.x = (int32_t)(first + rounding);

		first = -_sin * x + _cos * z;
		rounding = for_rounding * (first > 0 ? +1 : -1);
		res.z = (int32_t)(first + rounding);

		res.y = y;
		res.w = src->w;

		return res;
	}

	template<typename _Ty>
	_Ty rotateXZ_3(Point3* origin, _Ty* src, float radians)
	{
		float _sin = sinf(radians);
		float _cos = cosf(radians);
		float first;
		float rounding;

		_Ty res;

		int x = src->x - origin->x;
		int y = src->y - origin->y;
		int z = src->z - origin->z;

		first = _cos * x - _sin * z;
		rounding = for_rounding * (first > 0 ? +1 : -1);
		res.y = (int32_t)(first + rounding);

		first = _sin * y + _cos * y;
		rounding = for_rounding * (first > 0 ? +1 : -1);
		res.z = (int32_t)(first + rounding);


		res.x = x;
		res.w = src->w;

		return res;
	}










}