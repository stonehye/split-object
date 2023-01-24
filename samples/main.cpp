#include <iostream>
#include <opencv2/core/utils/logger.hpp>

#include "SplitObjectCreator.hpp"
#include "InputDataLoad.hpp"

using namespace Merge;

std::string InputDataLoader::rgb_ext = "%rgb.jpg";
std::string InputDataLoader::depth_ext = "%depth.png";
std::string InputDataLoader::depth_raw_ext = "%depth_raw.bin";
std::string InputDataLoader::confidence_ext = "%confidence.png";
std::string InputDataLoader::sailency_ext = "%sailency.png";
std::string InputDataLoader::object_mask_ext = "%final_mask.png";
std::string InputDataLoader::acceleration_ext = "%acceleration.txt";


std::tuple<std::string, std::string> read_target_name(std::string file_path)
{
	std::ifstream target_file;
	target_file.open(file_path);

	char arr[64];
	std::vector<std::string> accel_list;
	if (!target_file.is_open())
		return { "", "" };

	target_file.getline(arr, 64, '\n');
	std::string target = std::string(arr);
	target_file.getline(arr, 64, '\n');
	std::string filename = std::string(arr);

	target_file.close();

	return { target, filename };
}

int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	const fs::path root_dir = fs::current_path() / ".." / ".." / "samples" / "data";
	const fs::path target_containing_file = root_dir / "target_sample.txt";
	const auto [target, filename] = read_target_name(target_containing_file.generic_string());

	const fs::path target_root_dir = root_dir / target;
	const fs::path target_file_path = target_root_dir / filename;

	std::shared_ptr<InputData> data = std::make_shared<InputData>();
	InputDataLoader::Load_single(target_file_path.generic_string(), data);
	const auto& data_list = InputDataLoader::Load_all(target_root_dir.generic_string());

	const float fx = 1484.5f;
	const float fy = 1484.5f;
	const float cx = 951.96f;
	const float cy = 725.66f;

	SplitObject::Ptr split = create_split_object(false);

	//for (auto data : data_list)
	{
#ifdef _VIS
		data->Visualize();
#endif
		const auto& [label, mask_list] = split->Process(data->rgb, data->depth, data->confidence, data->object_mask);	
	}

	return 0;
}