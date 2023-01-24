
#ifndef common_h
#define common_h

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgcodecs/legacy/constants_c.h"

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include <numeric>
#include <unordered_map>
#include <vector>

using namespace std;

typedef vector<cv::Point>	VP;
typedef vector< VP >	VVP;
typedef unsigned int uint;

#define _INFINITY 1024


int inline sgn(float val) {
    return (0.f < val) - (val < 0.f);
};


bool inline isInf(float x)
{
	union
	{
		float f;
		int	  i;
	} u;

	u.f = x;
	u.i &= 0x7fffffff;
	return !(u.i ^ 0x7f800000);
};


float inline Slope(float x1, float y1, float x2, float y2)
{
	//reference slope
		float den = float(x2 - x1);
		float num = float(y2 - y1);
		if(den != 0)
		{
			return (num / den);
		}
		else
		{
			return ((num > 0) ? float(_INFINITY) : float(-_INFINITY));
		}
};

void Canny(cv::InputArray image, cv::OutputArray _edges,
           cv::OutputArray _sobel_x, cv::OutputArray _sobel_y,
                int apertureSize, bool L2gradient );


float inline ed2(const cv::Point& A, const cv::Point& B)
{
	return float(((B.x - A.x)*(B.x - A.x) + (B.y - A.y)*(B.y - A.y)));
}

void Labeling(cv::Mat1b& image, vector<vector<cv::Point> >& segments, int iMinLength);

bool SortBottomLeft2TopRight(const cv::Point& lhs, const cv::Point& rhs);
bool SortTopLeft2BottomRight(const cv::Point& lhs, const cv::Point& rhs);



float GetMinAnglePI(float alpha, float beta);

struct EllipseM
{
    float _xc;
    float _yc;
    float _a;
    float _b;
    float _rad;
    float _score;

    float _distance_from_center;
    
    EllipseM() : _xc(0.f), _yc(0.f), _a(0.f), _b(0.f), _rad(0.f), _score(0.f), _distance_from_center(0.f) {};
    EllipseM(float xc, float yc, float a, float b, float rad, float score = 0.f, float dist_from_center = 0.f) : _xc(xc), _yc(yc), _a(a), _b(b), _rad(rad), _score(score), _distance_from_center(dist_from_center) {};
    EllipseM(const EllipseM& other) : _xc(other._xc), _yc(other._yc), _a(other._a), _b(other._b), _rad(other._rad), _score(other._score), _distance_from_center(other._distance_from_center) {};

    void Draw(cv::Mat& img, const cv::Scalar& color, const int thickness)
    {
        ellipse(img, cv::Point(cvRound(_xc),cvRound(_yc)), cv::Size(cvRound(_a),cvRound(_b)), _rad * 180.0 / CV_PI, 0.0, 360.0, color, thickness);
    };

    void Draw(cv::Mat3b& img, const int thickness)
    {
        cv::Scalar color(0, cvFloor(255.f * _score), 0);
        ellipse(img, cv::Point(cvRound(_xc),cvRound(_yc)), cv::Size(cvRound(_a),cvRound(_b)), _rad * 180.0 / CV_PI, 0.0, 360.0, color, thickness);
    };
    
    void Set_distance_from_center(const int rgb_height, const int rgb_center_x, const int rgb_center_y)
    {
        int x_diff = (rgb_height - rgb_center_y) - (int)(_xc + 0.5f), y_diff = rgb_center_x - (int)(_yc + 0.5f);
        _distance_from_center = sqrt(x_diff * x_diff + y_diff * y_diff);
    }
    
    bool operator<(const EllipseM& other) const
    {
        if(_score == other._score)
        {
            float lhs_e = _b / _a;
            float rhs_e = other._b / other._a;
            if(lhs_e == rhs_e)
            {
                return false;
            }
            return lhs_e > rhs_e;
        }
        return _score > other._score;
    };
};

struct EllipseDistanceFromCenterCmp
{
    bool operator()(const EllipseM& a,
        const EllipseM& b) const {
        return a._distance_from_center < b._distance_from_center;
    }
};

#endif
