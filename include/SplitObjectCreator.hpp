
#ifndef SplitObjectCreator_hpp
#define SplitObjectCreator_hpp

#include "SplitFood.hpp"

namespace Merge
{
	SplitObject::Ptr create_split_object(bool is_food)
	{
		if (is_food)
			return std::make_shared<SplitFood>();

		return std::make_shared<SplitObject>();
	}
}

#endif