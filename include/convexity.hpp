#ifndef Convexity_hpp
#define Convexity_hpp

#include "cvutils.hpp"

namespace Merge
{
	cv::Mat compute_normal_map(const cv::Mat& depth, const int interval = 2)
	{
		const int width = depth.cols - (interval << 1);
		const int height = depth.rows - (interval << 1);

		cv::Mat prev_x = depth(cv::Rect(0, interval, width, height));
		cv::Mat next_x = depth(cv::Rect(interval << 1, interval, width, height));
		cv::Mat prev_y = depth(cv::Rect(interval, 0, width, height));
		cv::Mat next_y = depth(cv::Rect(interval, interval << 1, width, height));

		cv::Mat dzdx = (next_x - prev_x) / 2.0;
		cv::Mat dzdy = (next_y - prev_y) / 2.0;
		cv::Mat ones = cv::Mat::ones(next_x.size(), CV_32FC1);

		cv::Mat sq_sum = dzdx.mul(dzdx) + dzdy.mul(dzdy) + ones;
		cv::Mat norm;
		cv::sqrt(sq_sum, norm);
		norm += 1e-15;

		cv::Mat merged[3] = { dzdy.mul(1 / norm), dzdx.mul(1 / norm), ones.mul(1 / norm) };
		cv::Mat normal_map;

		cv::merge(merged, 3, normal_map);

		return normal_map;
	}

	float compute_convexity(const cv::Vec3f n1, const cv::Vec3f c1, const cv::Vec3f n2, const cv::Vec3f c2)
	{
		cv::Vec3f c = c1 - c2;
		c /= (cv::norm(c) + 1e-15);

		float cos1 = n1.dot(c);
		float cos2 = n2.dot(c);

		return cos1 - cos2;//(cos1 - cos2 + 2) / 4.0;
	}

	cv::Mat compute_convexity_map(const cv::Mat& normal_map, const cv::Mat& depth, const cv::Mat& mask, const int window_size = 5)
	{
		cv::Mat convexity_map = cv::Mat::zeros(normal_map.rows, normal_map.cols, CV_32FC1);
		const int half_window_size = window_size >> 1;

		for (int i = half_window_size; i < normal_map.rows - half_window_size; i++)
		{
			for (int j = half_window_size; j < normal_map.cols - half_window_size; j++)
			{
				if (mask.at<unsigned char>(i, j) == 0) continue;

				float convexity_sum = 0.0;
				for (int m = i - half_window_size; m <= i + half_window_size; m++)
				{
					for (int n = j - half_window_size; n <= j + half_window_size; n++)
					{
						convexity_sum += compute_convexity(normal_map.at<cv::Vec3f>(i, j), cv::Vec3f(i, j, depth.at<float>(i, j)),
							normal_map.at<cv::Vec3f>(m, n), cv::Vec3f(m, n, depth.at<float>(m, n)));
					}
				}
				convexity_map.at<float>(i, j) = convexity_sum / (window_size * window_size);
			}
		}

		return convexity_map;
	}

	cv::Mat z_normalize(const cv::Mat& mat, const cv::Mat& mask, const float mean, const float std, const int max=255)
	{
		cv::Mat result;

		result = (mat - mean) / std * max;

		result.setTo(max, result >= max);
		result.setTo(0, result <= 0);
		result.setTo(0, ~mask);
		
		result.convertTo(result, CV_8UC1);

		return result;
	}

	cv::Mat compute_variance_map(const cv::Mat& depth, const cv::Mat& mask, const int window_size=3)
	{
		const int half_ws = window_size >> 1;
		cv::Mat sigma = cv::Mat::zeros(depth.size(), CV_32FC1);

		for (int i = half_ws; i < depth.rows - half_ws; i++)
		{
			for (int j = half_ws; j < depth.cols - half_ws; j++)
			{
				if (mask.at<unsigned char>(i, j) == 0) continue;

				cv::Scalar mean, std;
				cv::meanStdDev(depth(cv::Rect(j - half_ws, i - half_ws, window_size, window_size)), mean, std);
				sigma.at<float>(i, j) = std[0]*std[0];
			}
		}

		sigma.setTo(255, sigma >= 255);
		sigma.setTo(0, sigma < 0);
		sigma.convertTo(sigma, CV_8UC1);

		sigma.setTo(255, sigma > cv::mean(sigma, mask)[0]);
		sigma.setTo(0, sigma < cv::mean(sigma, mask)[0]);
		
		return sigma;
	}

	cv::Mat extract_concave_area(const cv::Mat& convexity_map, const cv::Mat& mask, const float convexity_mean, const float convexity_std)
	{
		cv::Scalar mean, stddev;
		cv::meanStdDev(convexity_map, mean, stddev, mask);

		cv::Mat concave_area;
		cv::bitwise_and(convexity_map < (mean[0] - 1.2*stddev[0]), mask, concave_area);

		return concave_area;
	}
}
#endif