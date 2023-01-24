#include "InputDataLoad.hpp"

using namespace Merge;

void test_load_input_data()
{

	const fs::path data_root_dir = fs::current_path() / "data" / "adjacent object";
	const std::string filename = "2021-02-25_16=20=59=822";
	const fs::path file_path = data_root_dir / filename;

	/* Example of loading single input data with file name */
	std::shared_ptr<InputData> data = std::make_shared<InputData>();
	InputDataLoader::Load_single(file_path.generic_string(), data);
	std::cout << data->file_name << std::endl;

	/* Example of loading all input data in root directory */
	//auto data_list = InputDataLoader::Load_all(data_root_dir.generic_string());

	//for (auto& data : data_list)
	//{
	//	//access input data, and do something
	//	std::cout << data.file_name << std::endl;
	//}

}