#ifndef Stats2D_h
#define Stats2D_h

#include <iostream>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <numeric>

namespace Merge
{
	struct Stats2D
	{
		double sx, sy, sxx, syy, sxy;
		int N;

		Stats2D() : sx(0), sy(0),
			sxx(0), syy(0),
			sxy(0), N(0) {}

		Stats2D(const Stats2D& a, const Stats2D& b) :
			sx(a.sx + b.sx), sy(a.sy + b.sy),
			sxx(a.sxx + b.sxx), syy(a.syy + b.syy),
			sxy(a.sxy + b.sxy), N(a.N + b.N) {}

		inline void clear() 
		{
			sx = sy = sxx = syy = sxy = 0;
			N = 0;
		}

		inline void push(const double x, const double y) 
		{
			sx += x; sy += y;
			sxx += x * x; syy += y * y;
			sxy += x * y;
			++N;
		}

		inline void push(const Stats2D& other) 
		{
			sx += other.sx; sy += other.sy;
			sxx += other.sxx; syy += other.syy;
			sxy += other.sxy;
			N += other.N;
		}

		inline void pop(const double x, const double y) 
		{
			sx -= x; sy -= y;
			sxx -= x * x; syy -= y * y;
			sxy -= x * y;
			--N;

			assert(N >= 0);
		}

		inline void pop(const Stats2D& other) 
		{
			sx -= other.sx; sy -= other.sy;
			sxx -= other.sxx; syy -= other.syy;
			sxy -= other.sxy;
			N -= other.N;

			assert(N >= 0);
		}

		inline void compute(double center[2], double eigenvalue[2], double eigenvector[2][2]) const
		{
			assert(N >= 4);

			const double sc = ((double)1.0) / this->N;

			center[0] = sx * sc;
			center[1] = sy * sc;

			double K[2][2] = {
				{sxx - sx * sx * sc, sxy - sx * sy * sc},
				{				  0, syy - sy * sy * sc}
			};
			K[1][0] = K[0][1];
			double sv[2] = { 0,0 };
			double V[2][2] = { 0 };

			Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(
				Eigen::Map<Eigen::Matrix2d>(K[0], 2, 2));
			Eigen::Map<Eigen::Vector2d>(sv, 2, 1) = es.eigenvalues();
			//below we need to specify row major since V!=V'
			Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>>(V[0], 2, 2) = es.eigenvectors();

			eigenvalue[0] = sv[0];
			eigenvalue[1] = sv[1];

			if (V[0][0] * center[0] + V[1][0] * center[1] <= 0) 
			{
				eigenvector[0][0] = V[0][0];
				eigenvector[1][0] = V[1][0];
			}
			else 
			{
				eigenvector[0][0] = -V[0][0];
				eigenvector[1][0] = -V[1][0];
			}

			if (V[0][1] * center[0] + V[1][1] * center[1] <= 0) 
			{
				eigenvector[0][1] = V[0][1];
				eigenvector[1][1] = V[1][1];
			}
			else 
			{
				eigenvector[0][1] = -V[0][1];
				eigenvector[1][1] = -V[1][1];
			}
		}
	};

	struct ConvexityStats2D : public Stats2D
	{
		float convexity;
		int width, height, left, top;
		int center_x, center_y;
		Eigen::Vector2f long_axis, short_axis;
		float long_var, short_var, ratio;
		
		Eigen::Vector2f pos_end_point, neg_end_point;
		cv::Mat concave_line_path;

		ConvexityStats2D(int w, int h, int l, int t, int cx, int cy) :
			convexity(0), long_var(0), short_var(0), ratio(0),
			width(w), height(h), left(l), top(t), center_x(cx), center_y(cy) {}

		void clear()
		{
			Stats2D::clear();
			convexity = 0;
			width = height = left = top = center_x = center_y = 0;
			long_axis = short_axis = Eigen::Vector2f(0, 0);
			long_var = short_var = ratio = 0;
		}

		void push(float convex, int x, int y)
		{
			Stats2D::push(x, y);
			convexity += convex;
		}

		void compute(const cv::Mat& mask)
		{
			double center[2];
			double eigenvalue[2];
			double eigenvector[2][2];

			Stats2D::compute(center, eigenvalue, eigenvector);

			convexity /= N;

			long_axis = Eigen::Vector2f(eigenvector[0][1], eigenvector[1][1]);
			short_axis = Eigen::Vector2f(eigenvector[0][0], eigenvector[1][0]);

			long_var = eigenvalue[1];
			short_var = eigenvalue[0];
			ratio = eigenvalue[0] / eigenvalue[1];

			concave_line_path = cv::Mat::zeros(mask.size(), CV_8UC1);
			pos_end_point = Eigen::Vector2f(center_x, center_y);
			neg_end_point = Eigen::Vector2f(center_x, center_y);

			Eigen::Vector2f step = long_axis * 0.5;

			while (mask.at<unsigned char>(pos_end_point.y(), pos_end_point.x()) == 255)
			{
				concave_line_path.at<unsigned char>(pos_end_point.y(), pos_end_point.x()) = 255;
				pos_end_point += step;
			}

			while (mask.at<unsigned char>(neg_end_point.y(), neg_end_point.x()) == 255)
			{
				concave_line_path.at<unsigned char>(neg_end_point.y(), neg_end_point.x()) = 255;
				neg_end_point -= step;
			}
		}

		bool is_elongated()
		{
			return ratio < 0.05;
		}
	};
}
#endif
