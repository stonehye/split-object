
#ifndef SplitObject_hpp
#define SplitObject_hpp

#include "convexity.hpp"
#include "Stats2D.hpp"

namespace Merge
{
	class SplitObject
	{
	public:
		typedef std::shared_ptr<SplitObject> Ptr;
		typedef std::pair<cv::Mat, std::vector<std::pair<int, cv::Mat>>> Result;
		
	public:
		virtual Result Process(const cv::Mat& rgb, const cv::Mat& depth, const cv::Mat& confidence, const cv::Mat& object_mask)
		{
			Reset();
			_Preprocess_input(depth, object_mask);

			_normal_map = compute_normal_map(_depth(_object_rect));

			_convexity_map = compute_convexity_map(_normal_map, _depth(_normal_map_rect), _object_mask);

			_Extract_mask_contour();

			cv::Scalar mean, stddev;
			cv::meanStdDev(_convexity_map, mean, stddev, _object_mask);

			_concave_area = extract_concave_area(_convexity_map, _object_mask, mean[0], stddev[0]);
			cv::bitwise_and(_concave_area, ~_thick_contour_img, _concave_area, _object_mask);

			_variance_map = compute_variance_map(_depth(_normal_map_rect), _object_mask);
			cv::bitwise_and(_variance_map, ~_thick_contour_img, _high_variacne_area, _object_mask);

			auto concave_area_stats = _Compute_area_stats(_concave_area);
			auto var_area_stats = _Compute_area_stats(_high_variacne_area);

			_extended_concave_area = _Extend_area(concave_area_stats, _thin_contour_img);
			_extended_variance_area = _Extend_area(var_area_stats, _thin_contour_img);

			cv::Mat area_chosen = (concave_area_stats.size() > 0) ? _extended_concave_area : _extended_variance_area;
			_Finalize(area_chosen);

#ifdef _DEBUG
			printf("Split object number : %d\n", _split_object_mask_list.size());
#endif
#ifdef _VIS
			_z_normalized_convexity_map = z_normalize(_convexity_map, _object_mask, mean[0], stddev[0]);
			_Visualize();
#endif
			return { _object_label, _split_object_mask_list };
		}

		void Reset()
		{
			_Reset_input();
			_Reset_intermediate_output();
			_Reset_output();
		}

	private:
		virtual void _Reset_input()
		{
			_depth.release();
			_object_mask.release();
			_object_rect = _normal_map_rect = cv::Rect();
		}

		virtual void _Reset_intermediate_output()
		{
			_normal_map.release();
			_convexity_map.release();
			_z_normalized_convexity_map.release();
			_variance_map.release();

			_thick_contour_img.release();
			_thin_contour_img.release();

			_concave_area.release();
			_high_variacne_area.release();

			_extended_concave_area.release();
			_extended_variance_area.release();
		}

		void _Reset_output()
		{
			_object_label.release();
			_split_object_mask_list.clear();
		}

		void _Preprocess_input(const cv::Mat& depth, const cv::Mat& object_mask, const int normal_map_interval=2)
		{
			if (_depth.type() == CV_32FC1)
				depth.clone().convertTo(_depth, CV_32FC1, 1000.0);
			else
				depth.clone().convertTo(_depth, CV_32FC1);

			_object_rect = cv::boundingRect(object_mask);
			_object_rect = expand_rect(_object_rect, 10, 20, object_mask.cols, object_mask.rows);
			_normal_map_rect = expand_rect(_object_rect, -1 * (normal_map_interval << 1), -1 * (normal_map_interval << 1), object_mask.cols, object_mask.rows);

			_object_mask = object_mask.clone()(_normal_map_rect);
		}

		void _Extract_mask_contour()
		{
			std::vector<std::vector<cv::Point> > contours;
			std::vector<cv::Vec4i> hierarchy;

			_thick_contour_img = cv::Mat::zeros(_convexity_map.size(), CV_8UC1);
			_thin_contour_img = cv::Mat::zeros(_convexity_map.size(), CV_8UC1);

			cv::findContours(_object_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			cv::drawContours(_thick_contour_img, contours, 0, cv::Scalar(255, 255, 255), 5);
			cv::drawContours(_thin_contour_img, contours, 0, cv::Scalar(255, 255, 255), 2);
		}

		std::vector<std::pair<cv::Mat, ConvexityStats2D>> _Compute_area_stats(const cv::Mat& area, const float min_area_ratio=0.01)
		{
			cv::Mat label, label_stats, centroids;
			int label_num = cv::connectedComponentsWithStats(area, label, label_stats, centroids);

			const int object_area = area.rows * area.cols;
			std::vector<std::pair<cv::Mat, ConvexityStats2D>> result;

			for (int i = 1; i < label_num; i++)
			{
				int area = label_stats.at<int>(i, cv::CC_STAT_AREA);
				int width = label_stats.at<int>(i, cv::CC_STAT_WIDTH);
				int height = label_stats.at<int>(i, cv::CC_STAT_HEIGHT);
				int left = label_stats.at<int>(i, cv::CC_STAT_LEFT);
				int top = label_stats.at<int>(i, cv::CC_STAT_TOP);
				int center_x = centroids.at<double>(i, 0);
				int center_y = centroids.at<double>(i, 1);

				if (object_area * min_area_ratio > area)
					continue;

				ConvexityStats2D stats(width, height, left, top, center_x, center_y);

				cv::Mat label_mask = (label == i);
				for (int j = top; j < top + height; j++)
				{
					for (int k = left; k < left + width; k++)
					{
						if (label_mask.at<unsigned char>(j, k) != 255)
							continue;

						stats.push(0, k, j);
					}
				}
				stats.compute(label_mask);

				if (stats.is_elongated())
					result.push_back({ label_mask, stats });
			}

			return result;
		}

		cv::Mat _Extend_area(const std::vector<std::pair<cv::Mat, ConvexityStats2D>>& label_stats, const cv::Mat& contour_img, const float step=0.5)
		{
			cv::Mat extended = contour_img.clone();

			for (const auto& [label, stats] : label_stats)
			{
				Eigen::Vector2f ex_end1 = stats.pos_end_point, ex_end2 = stats.neg_end_point;
				Eigen::Vector2f next_step = stats.long_axis * step;
				cv::bitwise_or(extended, stats.concave_line_path, extended);

				while (contour_img.at<unsigned char>(ex_end1.y(), ex_end1.x()) != 255)
				{
					extended.at<unsigned char>(ex_end1.y(), ex_end1.x()) = 255;
					ex_end1 += next_step;
				}

				while (contour_img.at<unsigned char>(ex_end2.y(), ex_end2.x()) != 255)
				{
					extended.at<unsigned char>(ex_end2.y(), ex_end2.x()) = 255;
					ex_end2 -= next_step;
				}
			}

			return extended;
		}

		void _Finalize(const cv::Mat& area, const float min_area_ratio = 0.01)
		{
			const int object_area = area.rows * area.cols;
			cv::Mat stats, centers;

			cv::Mat area_clone = area.clone();

			area_clone.setTo(255, ~_object_mask);
			int object_num = cv::connectedComponentsWithStats(~area_clone, _object_label, stats, centers, 4);

			for (int i = 1; i < object_num; i++)
			{
				if (stats.at<int>(i, cv::CC_STAT_AREA) > object_area * min_area_ratio)
					_split_object_mask_list.push_back({ i, _object_label == i });
			}
		}

		void _Visualize()
		{
			cv::imshow("normal map", _normal_map);
			cv::imshow("convexity map", _z_normalized_convexity_map);
			cv::imshow("variance map", _variance_map);

			cv::imshow("mask thick contour", _thick_contour_img);
			cv::imshow("mask thin contour", _thin_contour_img);

			cv::imshow("concave area", _concave_area);
			cv::imshow("high variance area", _high_variacne_area);

			cv::imshow("extended concave area", _extended_concave_area);
			cv::imshow("extended variance area", _extended_variance_area);

			for (const auto& [id, mask] : _split_object_mask_list)
				cv::imshow("object " + std::to_string(id), mask);

			cv::waitKey(0);
			cv::destroyAllWindows();
		}

	protected:
		//input
		cv::Mat _rgb, _confidence;
		cv::Mat _depth, _object_mask;
	
		//final output
		cv::Mat _object_label;
		std::vector<std::pair<int, cv::Mat>> _split_object_mask_list;

	private:
		cv::Rect _object_rect, _normal_map_rect; // input mask rectangle

		//intermediate output
		cv::Mat _normal_map, _convexity_map, _z_normalized_convexity_map, _variance_map;
		cv::Mat _thick_contour_img, _thin_contour_img;
		cv::Mat _concave_area, _high_variacne_area;
		cv::Mat _extended_concave_area, _extended_variance_area;
	};
}

#endif