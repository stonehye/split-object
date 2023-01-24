
#ifndef SplitFood_hpp
#define SplitFood_hpp

#include "SplitObject.hpp"
#include "Shape/CNEllipseDetector.h"

namespace Merge
{
	class SplitFood : public SplitObject
	{
	public:
		Result Process(const cv::Mat& rgb, const cv::Mat& depth, const cv::Mat& confidence, const cv::Mat& object_mask) override
		{
			Reset();
			_Preprocess_input(rgb);

			_ellipse_detector.Detect(_gray, _ellipses, rgb.rows, rgb.cols >> 1, rgb.rows >> 1);

#ifdef _VIS
			if (_ellipses.size() > 0)
			{
				_ellipse_detector.DrawDetectedEllipses(_ellipse_result_vis, _ellipses);
				cv::imshow("ellipse result", _ellipse_result_vis);
				cv::waitKey(0);
			}
#endif
			return { _object_label, _split_object_mask_list };
		}

	private:
		virtual void _Reset_input()
		{
			_rgb.release();
			_depth.release();
			_confidence.release();
			_object_mask.release();

			_gray.release();
		}

		virtual void _Reset_intermediate_output()
		{
			_ellipses.clear();
			_ellipse_result_vis.release();
		}

		void _Preprocess_input(const cv::Mat& rgb)
		{
			cvtColor(rgb, _gray, cv::COLOR_BGR2GRAY);
			_ellipse_result_vis = rgb.clone();
		}

	private:
		CNEllipseDetector _ellipse_detector;

	private:
		cv::Mat1b _gray; // input gray

		std::vector<EllipseM> _ellipses;
		cv::Mat _ellipse_result_vis;
	};
}

#endif