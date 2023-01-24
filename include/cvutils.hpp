#ifndef cvutils_hpp
#define cvutils_hpp

#include <opencv2/opencv.hpp>

namespace Merge
{
    cv::Rect expand_rect(const cv::Rect& rect, const int width, const int height, const int max_width, const int max_height)
    {
        cv::Rect expanded_rect = rect;

        expanded_rect.x -= (width >> 1);
        expanded_rect.width += width;
        expanded_rect.y -= (height >> 1);
        expanded_rect.height += height;

        expanded_rect.width = (expanded_rect.x + expanded_rect.width >= max_width) ? max_width - expanded_rect.x : expanded_rect.width;
        expanded_rect.height = (expanded_rect.y + expanded_rect.height >= max_height) ? max_height - expanded_rect.y : expanded_rect.height;

        return expanded_rect;
    }
}
#endif